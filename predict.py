import os
import re
import json
import math
import time
import tempfile
from typing import Optional

# CRITICAL: force offline BEFORE importing transformers so from_pretrained never
# tries to reach an HF mirror at runtime (causes silent 615s boot timeout).
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
# Reduz fragmentação de memória CUDA (ajuda em vídeo com muitos frames).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
# Limita o orçamento TOTAL de pixels do vídeo (todos os frames somados). Sem flash-attn,
# a torre de visão usa SDPA, que materializa a matriz de atenção O(n²) ~ 16*seq²*2 bytes,
# com seq = pixels/784. O default do modelo (~22.6M px) estoura a L40S (OOM 26GiB).
# 8M px -> seq~10k -> matriz ~3GB, com folga. Lido no import de processing_locateanything.py.
os.environ.setdefault("VIDEO_MAX_PIXELS", str(8_000_000))

import torch
import cv2
import av
import numpy as np
from scipy.optimize import linear_sum_assignment
from PIL import Image, ImageDraw, ImageFont
from cog import BasePredictor, BaseModel, Input, Path
from transformers import AutoConfig, AutoModel, AutoTokenizer, AutoProcessor
from transformers import CLIPModel, CLIPProcessor

MODEL_PATH = "/src/weights/model"

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".mpeg", ".mpg", ".wmv", ".flv"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff", ".heic"}


def _looks_like_video(path):
    """Detecta vídeo vs imagem pela extensão (com fallback sniffando como imagem),
    pra rotear certo mesmo se o usuário subir o vídeo no campo 'image' ou vice-versa."""
    ext = os.path.splitext(str(path))[1].lower()
    if ext in VIDEO_EXTS:
        return True
    if ext in IMAGE_EXTS:
        return False
    try:
        Image.open(str(path)).verify()
        return False
    except Exception:
        return True

DEFAULT_PROMPT = "Detect all the main objects and output their bounding boxes."


def _parse_detections(answer: str, width, height):
    """Extract bounding boxes and points from the model's box-token output.

    Boxes:  <box><x1><y1><x2><y2></box>  (coords normalized to 0-1000)
    Points: <box><x><y></box>

    If width/height are given (single image), coords are rescaled to pixels.
    Otherwise (video, frames vary) coords are returned as 0-1 fractions.
    """
    def sx(v):
        return round(v / 1000 * width, 2) if width else round(v / 1000, 4)

    def sy(v):
        return round(v / 1000 * height, 2) if height else round(v / 1000, 4)

    boxes = []
    for m in re.finditer(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>", answer):
        x1, y1, x2, y2 = (int(g) for g in m.groups())
        boxes.append({"x1": sx(x1), "y1": sy(y1), "x2": sx(x2), "y2": sy(y2)})

    points = []
    for m in re.finditer(r"<box><(\d+)><(\d+)></box>", answer):
        x, y = int(m.group(1)), int(m.group(2))
        points.append({"x": sx(x), "y": sy(y)})

    return boxes, points


def _draw(pil_image, boxes, points):
    """Desenha as bounding boxes (com índice) e os points sobre a imagem."""
    img = pil_image.convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except Exception:
        font = ImageFont.load_default()
    color = (0, 230, 118)  # verde
    Wimg, Himg = img.size
    for i, b in enumerate(boxes, start=1):
        # Ordena os cantos (o modelo às vezes devolve box invertido) e clampa aos limites.
        x1, x2 = sorted((b["x1"], b["x2"]))
        y1, y2 = sorted((b["y1"], b["y2"]))
        x1 = max(0, min(x1, Wimg - 1)); x2 = max(0, min(x2, Wimg - 1))
        y1 = max(0, min(y1, Himg - 1)); y2 = max(0, min(y2, Himg - 1))
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        label = str(i)
        tb = draw.textbbox((0, 0), label, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        draw.rectangle([x1, y1, x1 + tw + 6, y1 + th + 6], fill=color)
        draw.text((x1 + 3, y1 + 1), label, fill=(0, 0, 0), font=font)
    for p in points:
        x, y = p["x"], p["y"]
        r = 6
        draw.ellipse([x - r, y - r, x + r, y + r], outline=color, width=3)
    return img


def _nms_boxes(boxes, iou_thr=0.45, contain_thr=0.80):
    """Remove caixas duplicadas (mesmas detectadas em tiles que se sobrepõem).
    Sem score do modelo -> ordena por área (mantém a mais completa) e suprime
    as muito sobrepostas (IoU) ou quase contidas (containment)."""
    arr = sorted(boxes, key=lambda b: (b["x2"] - b["x1"]) * (b["y2"] - b["y1"]), reverse=True)
    kept = []
    for b in arr:
        dup = False
        ab = max(1.0, (b["x2"] - b["x1"]) * (b["y2"] - b["y1"]))
        for k in kept:
            ix1 = max(b["x1"], k["x1"]); iy1 = max(b["y1"], k["y1"])
            ix2 = min(b["x2"], k["x2"]); iy2 = min(b["y2"], k["y2"])
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            ak = max(1.0, (k["x2"] - k["x1"]) * (k["y2"] - k["y1"]))
            iou = inter / (ab + ak - inter)
            if iou > iou_thr or (inter / ab) > contain_thr:
                dup = True
                break
        if not dup:
            kept.append(b)
    return kept


def _dedup_points(points, min_dist_frac=0.012, diag=1000.0):
    """Junta points muito próximos (mesmo objeto visto em tiles vizinhos)."""
    thr = (min_dist_frac * diag) ** 2
    kept = []
    for p in points:
        if all((p["x"] - q["x"]) ** 2 + (p["y"] - q["y"]) ** 2 > thr for q in kept):
            kept.append(p)
    return kept


def _iou_batch(dets, trks):
    """IoU entre cada det (N,4) e cada track (M,4) -> matriz (N,M)."""
    trks = np.expand_dims(trks, 0)
    dets = np.expand_dims(dets, 1)
    xx1 = np.maximum(dets[..., 0], trks[..., 0])
    yy1 = np.maximum(dets[..., 1], trks[..., 1])
    xx2 = np.minimum(dets[..., 2], trks[..., 2])
    yy2 = np.minimum(dets[..., 3], trks[..., 3])
    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    inter = w * h
    area_d = (dets[..., 2] - dets[..., 0]) * (dets[..., 3] - dets[..., 1])
    area_t = (trks[..., 2] - trks[..., 0]) * (trks[..., 3] - trks[..., 1])
    return inter / (area_d + area_t - inter + 1e-9)


def _bbox_to_z(b):
    w = b[2] - b[0]; h = b[3] - b[1]
    x = b[0] + w / 2.0; y = b[1] + h / 2.0
    s = w * h; r = w / (h + 1e-9)
    return np.array([x, y, s, r], dtype=float).reshape((4, 1))


def _z_to_bbox(x):
    # x é o estado do Kalman, shape (7,1) -> indexar [i,0] p/ obter escalar 0-dim
    # (numpy 2.x recusa float() de array 1-D).
    s = max(float(x[2, 0]), 1e-3); r = max(float(x[3, 0]), 1e-3)
    w = math.sqrt(s * r); h = s / w
    cx = float(x[0, 0]); cy = float(x[1, 0])
    return np.array([cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0])


class _KalmanBox:
    """Filtro de Kalman de velocidade constante p/ uma bbox (estado SORT clássico:
    [cx, cy, s(area), r(aspect), vcx, vcy, vs])."""
    def __init__(self, bbox, track_id):
        self.id = track_id
        self.F = np.eye(7)
        self.F[0, 4] = self.F[1, 5] = self.F[2, 6] = 1.0
        self.H = np.zeros((4, 7))
        self.H[0, 0] = self.H[1, 1] = self.H[2, 2] = self.H[3, 3] = 1.0
        self.R = np.eye(4); self.R[2:, 2:] *= 10.0
        self.P = np.eye(7); self.P[4:, 4:] *= 1000.0; self.P *= 10.0
        self.Q = np.eye(7); self.Q[4:, 4:] *= 0.01; self.Q[-1, -1] *= 0.01
        self.x = np.zeros((7, 1)); self.x[:4] = _bbox_to_z(bbox)
        self.time_since_update = 0
        self.hits = 0
        self.feat = None        # embedding de aparência (ReID), média móvel

    def update_feat(self, f):
        if self.feat is None:
            self.feat = f
        else:
            self.feat = 0.9 * self.feat + 0.1 * f
            n = np.linalg.norm(self.feat) + 1e-9
            self.feat = self.feat / n

    def predict(self):
        if self.x[6] + self.x[2] <= 0:
            self.x[6] = 0.0
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.time_since_update += 1
        return _z_to_bbox(self.x)

    def update(self, bbox):
        z = _bbox_to_z(bbox)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(7) - K @ self.H) @ self.P
        self.time_since_update = 0
        self.hits += 1

    def state(self):
        return _z_to_bbox(self.x)


def _associate(dets, trks, iou_threshold):
    if len(trks) == 0:
        return np.empty((0, 2), dtype=int), list(range(len(dets))), []
    iou = _iou_batch(dets, trks)
    if min(iou.shape) > 0:
        a = (iou > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched = np.stack(np.where(a), axis=1)
        else:
            r, c = linear_sum_assignment(-iou)
            matched = np.array(list(zip(r, c)))
    else:
        matched = np.empty((0, 2), dtype=int)

    un_dets = [d for d in range(len(dets)) if d not in matched[:, 0]] if len(matched) else list(range(len(dets)))
    un_trks = [t for t in range(len(trks)) if t not in matched[:, 1]] if len(matched) else list(range(len(trks)))
    good = []
    for m in matched:
        if iou[m[0], m[1]] < iou_threshold:
            un_dets.append(m[0]); un_trks.append(m[1])
        else:
            good.append(m.reshape(1, 2))
    good = np.concatenate(good, axis=0) if good else np.empty((0, 2), dtype=int)
    return good, un_dets, un_trks


class _Sort:
    """SORT (Kalman + Hungarian), com modo ReID opcional. step() é chamado a CADA
    frame de vídeo; dets=None nos frames sem detecção (só prediz -> caixas suaves).
    No modo ReID, o custo de associação combina movimento (IoU) + aparência (cosine
    do embedding), permitindo re-identificar objetos após oclusão/cruzamento."""
    def __init__(self, max_age, draw_window, iou_threshold=0.2, reid=False, max_cos=0.4):
        self.max_age = max_age
        self.draw_window = draw_window
        self.iou_threshold = iou_threshold
        self.reid = reid
        self.max_cos = max_cos
        self.trackers = []
        self.next_id = 1

    def step(self, dets, feats=None):
        trks = np.zeros((len(self.trackers), 4))
        to_del = []
        for t, trk in enumerate(self.trackers):
            pos = trk.predict()
            trks[t] = pos
            if np.any(np.isnan(pos)):
                to_del.append(t)
        for t in reversed(to_del):
            self.trackers.pop(t)
        if to_del:
            trks = np.delete(trks, to_del, axis=0)

        if dets is not None and len(dets) > 0:
            assigned_d = set()
            if len(self.trackers) > 0:
                iou = _iou_batch(dets, trks)                      # (D,T)
                cosd = None
                if self.reid and feats is not None and all(t.feat is not None for t in self.trackers):
                    trk_feats = np.array([t.feat for t in self.trackers])   # (T,512)
                    cosd = 1.0 - feats @ trk_feats.T              # distância cosseno [0,2]
                    cost = 0.5 * (1.0 - iou) + 0.5 * (cosd / 2.0)
                else:
                    cost = 1.0 - iou
                r, c = linear_sum_assignment(cost)
                for di, ti in zip(r, c):
                    ok = iou[di, ti] >= self.iou_threshold
                    if self.reid and cosd is not None:
                        ok = ok or (cosd[di, ti] <= self.max_cos)
                    if ok:
                        self.trackers[ti].update(dets[di])
                        if self.reid and feats is not None:
                            self.trackers[ti].update_feat(feats[di])
                        assigned_d.add(di)
            for di in range(len(dets)):
                if di in assigned_d:
                    continue
                k = _KalmanBox(dets[di], self.next_id)
                self.next_id += 1
                if self.reid and feats is not None:
                    k.update_feat(feats[di])
                self.trackers.append(k)

        ret, keep = [], []
        for trk in self.trackers:
            if trk.time_since_update <= self.max_age:
                keep.append(trk)
            if trk.time_since_update <= self.draw_window:
                ret.append((trk.id, tuple(trk.state())))
        self.trackers = keep
        return ret


class Output(BaseModel):
    detections: str               # JSON com answer + boxes + points
    num_detections: int
    image: Optional[Path] = None  # imagem anotada (input de imagem)
    video: Optional[Path] = None  # vídeo anotado em H.264 (input de vídeo)


class Predictor(BasePredictor):
    def setup(self):
        t0 = time.time()
        self.device = "cuda"
        self.dtype = torch.bfloat16

        print(f"[setup] MODEL_PATH={MODEL_PATH}", flush=True)
        try:
            print(f"[setup] dir: {sorted(os.listdir(MODEL_PATH))[:30]}", flush=True)
        except Exception as e:
            print(f"[setup] cannot list MODEL_PATH: {e}", flush=True)
        print(f"[setup] cuda={torch.cuda.is_available()} devices={torch.cuda.device_count()}", flush=True)

        print(f"[setup] loading tokenizer... (t={time.time()-t0:.1f}s)", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH, trust_remote_code=True, local_files_only=True
        )
        print(f"[setup] loading processor... (t={time.time()-t0:.1f}s)", flush=True)
        self.processor = AutoProcessor.from_pretrained(
            MODEL_PATH, trust_remote_code=True, local_files_only=True
        )
        # Força SDPA em todo o modelo. O config vem com _attn_implementation="magi"
        # (MagiAttention, exclusivo Hopper/Blackwell). A L40S é Ada; o caminho magi
        # exige o pacote magi_attention (ausente). Zeramos qualquer chance de cair
        # no build_magi_ranges forçando sdpa em config/texto/visão.
        print(f"[setup] loading config (forçando sdpa)... (t={time.time()-t0:.1f}s)", flush=True)
        config = AutoConfig.from_pretrained(
            MODEL_PATH, trust_remote_code=True, local_files_only=True
        )
        config._attn_implementation = "sdpa"
        if hasattr(config, "text_config"):
            # texto em sdpa: evita o caminho 'magi' (MagiAttention, só Hopper/Blackwell).
            config.text_config._attn_implementation = "sdpa"
        if hasattr(config, "vision_config"):
            # visão em sdpa (flash-attn indisponível neste torch). O OOM de vídeo é
            # contido via VIDEO_MAX_PIXELS (ver topo do arquivo).
            config.vision_config._attn_implementation = "sdpa"

        print(f"[setup] loading model... (t={time.time()-t0:.1f}s)", flush=True)
        self.model = AutoModel.from_pretrained(
            MODEL_PATH,
            config=config,
            trust_remote_code=True,
            local_files_only=True,
            torch_dtype=self.dtype,
            attn_implementation="sdpa",
        ).to(self.device)
        self.model.eval()
        self._reid = None  # extrator de aparência ReID (lazy: só carrega se usado)
        self._clip = None  # (modelo, processor) CLIP p/ verificação (lazy)
        print(f"[setup] DONE (t={time.time()-t0:.1f}s)", flush=True)

    def _get_clip(self):
        if self._clip is None:
            m = CLIPModel.from_pretrained("/src/weights/clip", local_files_only=True)
            p = CLIPProcessor.from_pretrained("/src/weights/clip", local_files_only=True)
            self._clip = (m.eval().to(self.device), p)
        return self._clip

    @torch.no_grad()
    def _verify(self, pil, boxes, obj, threshold):
        """Verifica cada caixa com CLIP (zero-shot). MANTÉM a caixa a menos que o CLIP
        esteja confiante que é pedra/grama/fundo — i.e., descarta só se P('{obj}') < threshold.
        Threshold baixo = lenient (só corta falso-positivo óbvio). Alto = severo."""
        if not boxes:
            return boxes
        model, proc = self._get_clip()
        texts = [
            f"a photo of a {obj}",
            "a photo of a rock or stone",
            "a photo of grass, dirt or bare ground",
            "an empty blurry background",
        ]
        crops = []
        for b in boxes:
            x1 = max(0, int(min(b["x1"], b["x2"]))); x2 = max(x1 + 1, int(max(b["x1"], b["x2"])))
            y1 = max(0, int(min(b["y1"], b["y2"]))); y2 = max(y1 + 1, int(max(b["y1"], b["y2"])))
            crops.append(pil.crop((x1, y1, x2, y2)))
        kept = []
        B = 64
        for i in range(0, len(crops), B):
            chunk = crops[i:i + B]
            inputs = proc(text=texts, images=chunk, return_tensors="pt", padding=True).to(self.device)
            probs = model(**inputs).logits_per_image.softmax(dim=1)  # (n_crops, n_texts)
            pos = probs[:, 0].cpu().tolist()  # P('{obj}')
            for j, p in enumerate(pos):
                if p >= threshold:
                    kept.append(boxes[i + j])
        return kept

    def _get_reid(self):
        if self._reid is None:
            import torchvision
            m = torchvision.models.resnet18(weights=None)
            m.fc = torch.nn.Identity()
            sd = torch.load("/src/weights/reid_resnet18.pth", map_location="cpu")
            m.load_state_dict(sd, strict=False)
            self._reid = m.eval().to(self.device, dtype=self.dtype)
        return self._reid

    @torch.no_grad()
    def _embed(self, frame_rgb, boxes_xyxy):
        """Embedding de aparência (ResNet18) por caixa. Retorna (N,512) L2-normalizado."""
        reid = self._get_reid()
        H, W = frame_rgb.shape[:2]
        crops = []
        for x1, y1, x2, y2 in boxes_xyxy:
            x1 = max(0, min(int(x1), W - 1)); x2 = max(0, min(int(x2), W))
            y1 = max(0, min(int(y1), H - 1)); y2 = max(0, min(int(y2), H))
            if x2 - x1 < 2 or y2 - y1 < 2:
                crop = np.zeros((128, 128, 3), dtype=np.uint8)
            else:
                crop = cv2.resize(frame_rgb[y1:y2, x1:x2], (128, 128))
            crops.append(crop)
        t = torch.from_numpy(np.stack(crops)).permute(0, 3, 1, 2).float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        t = (t - mean) / std
        feats = reid(t.to(self.device, dtype=self.dtype)).float()
        feats = torch.nn.functional.normalize(feats, dim=1)
        return feats.cpu().numpy()

    @torch.no_grad()
    def _infer_image(self, pil, prompt, generation_mode, max_new_tokens,
                     temperature, top_p, repetition_penalty):
        """Roda o grounding numa única imagem PIL e devolve a resposta crua (string)."""
        messages = [{
            "role": "user",
            "content": [{"type": "image", "image": pil}, {"type": "text", "text": prompt}],
        }]
        text = self.processor.py_apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        images, videos = self.processor.process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=images, videos=videos, return_tensors="pt"
        ).to(self.device)
        do_sample = temperature > 0.0
        response = self.model.generate(
            pixel_values=inputs["pixel_values"].to(self.dtype),
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            image_grid_hws=inputs.get("image_grid_hws", None),
            tokenizer=self.tokenizer,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            generation_mode=generation_mode,
            temperature=temperature if do_sample else 1.0,
            do_sample=do_sample,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            verbose=False,
        )
        answer = response[0] if isinstance(response, tuple) else response
        return answer if isinstance(answer, str) else str(answer)

    @torch.no_grad()
    def _infer_tiled(self, pil, prompt, gen, n, overlap=0.15):
        """Tiling: divide a imagem em n×n pedaços (com sobreposição), detecta cada um
        em resolução cheia, mapeia as caixas de volta p/ a imagem inteira e dedup com NMS.
        Recupera objetos pequenos/ao fundo que o teto de resolução do modelo perderia."""
        W, H = pil.size
        tw, th = W / n, H / n
        ov_w, ov_h = tw * overlap, th * overlap
        all_boxes, all_points = [], []
        for r in range(n):
            for c in range(n):
                x0 = max(0, int(c * tw - ov_w)); y0 = max(0, int(r * th - ov_h))
                x1 = min(W, int((c + 1) * tw + ov_w)); y1 = min(H, int((r + 1) * th + ov_h))
                crop = pil.crop((x0, y0, x1, y1))
                cw, ch = crop.size
                ans = self._infer_image(crop, prompt, *gen)
                bxs, pts = _parse_detections(ans, cw, ch)  # pixels locais do tile
                for b in bxs:
                    all_boxes.append({"x1": b["x1"] + x0, "y1": b["y1"] + y0,
                                      "x2": b["x2"] + x0, "y2": b["y2"] + y0})
                for p in pts:
                    all_points.append({"x": p["x"] + x0, "y": p["y"] + y0})
        diag = (W ** 2 + H ** 2) ** 0.5
        return _nms_boxes(all_boxes), _dedup_points(all_points, diag=diag)

    @torch.no_grad()
    def predict(
        self,
        image: Path = Input(
            description="Imagem RGB de entrada. Use ISTO para grounding em imagem.",
            default=None,
        ),
        video: Path = Input(
            description="Vídeo de entrada. Use ISTO para grounding em vídeo "
                        "(amostrado em frames). Informe image OU video, não os dois.",
            default=None,
        ),
        prompt: str = Input(
            description="Instrução em linguagem natural do que localizar (ex: 'Detect the red car').",
            default=DEFAULT_PROMPT,
        ),
        generation_mode: str = Input(
            description="Modo de decodificação. 'hybrid' equilibra velocidade e precisão.",
            choices=["hybrid", "ar", "mtp"],
            default="hybrid",
        ),
        tiles: int = Input(
            description="[Imagem] Tiling: divide a imagem em NxN pedaços e detecta cada um "
                        "em resolução cheia (recupera objetos pequenos/ao fundo). 1 = desligado. "
                        "Ex.: 3 = 3x3 = 9 inferências (mais lento). Ótimo para CONTAGEM densa.",
            default=1, ge=1, le=5,
        ),
        verify: bool = Input(
            description="[Imagem] Verifica cada caixa com CLIP e descarta falso-positivo "
                        "(ex.: pedra/mato marcado como objeto). Requer 'verify_object'.",
            default=False,
        ),
        verify_object: str = Input(
            description="[Imagem] O que cada caixa DEVE ser, em inglês e no singular "
                        "(ex.: 'cow', 'egg', 'box', 'person'). Usado pela verificação CLIP.",
            default="",
        ),
        verify_threshold: float = Input(
            description="[Imagem] Severidade da verificação. Baixo (0.1-0.2) = lenient, só "
                        "corta falso-positivo ÓBVIO (recomendado). Alto (0.4+) = severo, "
                        "mas pode cortar objeto pequeno/borrado de verdade.",
            default=0.15, ge=0.0, le=0.9,
        ),
        detect_fps: float = Input(
            description="[Vídeo] Quantas vezes por segundo rodar a detecção. Mais alto = "
                        "caixas acompanham melhor o movimento, porém mais lento.",
            default=3.0, ge=0.5, le=10.0,
        ),
        max_detect_frames: int = Input(
            description="[Vídeo] Teto de frames detectados (distribuídos por todo o vídeo) "
                        "p/ limitar o tempo de processamento.",
            default=24, ge=1, le=120,
        ),
        tracker: str = Input(
            description="[Vídeo] 'none' = só caixas por frame, SEM ID (estilo NVIDIA, "
                        "sem renumeração). 'sort' = IDs por movimento (Kalman+Hungarian). "
                        "'reid' = SORT + aparência (ResNet18): IDs robustos em oclusão, mais lento.",
            choices=["none", "sort", "reid"],
            default="none",
        ),
        max_new_tokens: int = Input(default=2048, ge=64, le=8192),
        temperature: float = Input(default=0.2, ge=0.0, le=2.0),
        top_p: float = Input(default=0.9, ge=0.0, le=1.0),
        repetition_penalty: float = Input(default=1.1, ge=1.0, le=2.0),
    ) -> Output:
        if image is None and video is None:
            raise ValueError("Forneça 'image' OU 'video'.")

        gen = (generation_mode, max_new_tokens, temperature, top_p, repetition_penalty)

        # Roteia pelo TIPO REAL do arquivo, não pelo campo: se o usuário subir um vídeo
        # no campo 'image' (ou vice-versa), ainda funciona.
        src = str(video if video is not None else image)
        is_video = _looks_like_video(src)

        # ---------- IMAGEM ----------
        if not is_video:
            pil = Image.open(src).convert("RGB")
            width, height = pil.size
            if tiles and tiles > 1:
                # Tiling: NxN pedaços em resolução cheia + NMS -> recupera os pequenos.
                boxes, points = self._infer_tiled(pil, prompt, gen, int(tiles))
                answer = f"[tiled {tiles}x{tiles}]"
            else:
                answer = self._infer_image(pil, prompt, *gen)
                boxes, points = _parse_detections(answer, width, height)

            # Verificação CLIP: descarta caixas que não parecem o objeto-alvo (pedra/mato).
            verified_info = None
            if verify and verify_object.strip():
                n_before = len(boxes)
                boxes = self._verify(pil, boxes, verify_object.strip(), float(verify_threshold))
                verified_info = {"object": verify_object.strip(), "threshold": float(verify_threshold),
                                 "before": n_before, "after": len(boxes)}

            detections = json.dumps({
                "modality": "image", "answer": answer, "coords": "pixels",
                "tiles": int(tiles), "verified": verified_info,
                "image_width": width, "image_height": height,
                "boxes": boxes, "points": points,
            }, ensure_ascii=False)

            annotated = _draw(pil, boxes, points)
            out_path = os.path.join(tempfile.mkdtemp(), "annotated.png")
            annotated.save(out_path)
            return Output(
                detections=detections,
                num_detections=len(boxes) + len(points),
                image=Path(out_path),
            )

        # ---------- VÍDEO (detecção frame-a-frame + tracking p/ ID estável) ----------
        # Cada frame amostrado é detectado como IMAGEM (caixas acompanham o movimento);
        # um tracker por IoU mantém o MESMO ID em cada objeto entre frames (sem renumerar).
        cap = cv2.VideoCapture(src)
        orig_fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        W2, H2 = W - (W % 2), H - (H % 2)

        step = max(1, int(round(orig_fps / max(0.1, detect_fps))))
        if total > 0 and (total / step) > max_detect_frames:
            step = max(1, math.ceil(total / max_detect_frames))
        color = (0, 230, 118)  # RGB verde

        out = av.open(os.path.join(tempfile.mkdtemp(), "annotated.mp4"), mode="w")
        out_video_path = out.name
        vstream = out.add_stream("libx264", rate=max(1, int(round(orig_fps))))
        vstream.width, vstream.height, vstream.pix_fmt = W2, H2, "yuv420p"

        # 'none' = só caixas por frame, sem ID (estilo NVIDIA). 'sort'/'reid' = rastreio.
        # No SORT: prediz a cada frame (movimento suave), atualiza nos frames detectados;
        # max_age = ~2 intervalos (sobrevive a 1 detecção perdida); draw_window = 1 intervalo.
        mode = tracker
        use_reid = (mode == "reid")
        sort = _Sort(max_age=2 * step, draw_window=step, iou_threshold=0.2, reid=use_reid) \
            if mode in ("sort", "reid") else None
        cur_points = []
        cur_boxes_none = []     # modo none: caixas do último frame detectado (mantidas entre frames)
        per_frame = []
        max_per_frame = 0
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)[:H2, :W2]
            drawn = []
            if i % step == 0:
                pil_f = Image.fromarray(rgb)
                ans = self._infer_image(pil_f, prompt, *gen)
                boxes, cur_points = _parse_detections(ans, W2, H2)
                dets = []
                for b in boxes:
                    x1, x2 = sorted((float(b["x1"]), float(b["x2"])))
                    y1, y2 = sorted((float(b["y1"]), float(b["y2"])))
                    dets.append([x1, y1, x2, y2])
                dets = np.array(dets, dtype=float).reshape(-1, 4)
                if mode == "none":
                    cur_boxes_none = [tuple(map(float, d)) for d in dets]
                    max_per_frame = max(max_per_frame, len(cur_boxes_none) + len(cur_points))
                    per_frame.append({"frame": i, "time": round(i / orig_fps, 2),
                                      "num": len(cur_boxes_none) + len(cur_points)})
                else:
                    feats = self._embed(rgb, dets) if (use_reid and len(dets) > 0) else None
                    drawn = sort.step(dets, feats)
                    per_frame.append({"frame": i, "time": round(i / orig_fps, 2),
                                      "ids": [tid for tid, _ in drawn]})
            elif mode != "none":
                drawn = sort.step(None)

            if mode == "none":
                for box in cur_boxes_none:
                    bx1 = max(0, min(int(box[0]), W2 - 1)); by1 = max(0, min(int(box[1]), H2 - 1))
                    bx2 = max(0, min(int(box[2]), W2 - 1)); by2 = max(0, min(int(box[3]), H2 - 1))
                    cv2.rectangle(rgb, (bx1, by1), (bx2, by2), color, 3)
            else:
                for tid, box in drawn:
                    bx1 = max(0, min(int(box[0]), W2 - 1)); by1 = max(0, min(int(box[1]), H2 - 1))
                    bx2 = max(0, min(int(box[2]), W2 - 1)); by2 = max(0, min(int(box[3]), H2 - 1))
                    cv2.rectangle(rgb, (bx1, by1), (bx2, by2), color, 3)
                    cv2.putText(rgb, f"#{tid}", (bx1, max(14, by1 - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
            for p in cur_points:
                cv2.circle(rgb, (int(p["x"]), int(p["y"])), 6, color, 3)
            for pkt in vstream.encode(av.VideoFrame.from_ndarray(rgb, format="rgb24")):
                out.mux(pkt)
            i += 1
        for pkt in vstream.encode():
            out.mux(pkt)
        out.close()
        cap.release()

        # 'none' não tem IDs -> reporta o pico de objetos simultâneos num frame.
        count = max_per_frame if mode == "none" else (sort.next_id - 1)
        detections = json.dumps({
            "modality": "video", "coords": "pixels",
            "tracker": mode,
            "count_meaning": "pico_simultaneo" if mode == "none" else "objetos_unicos",
            "frame_width": W2, "frame_height": H2,
            "detected_frames": len(per_frame), "step": step,
            "count": count,
            "per_frame": per_frame,
        }, ensure_ascii=False)

        return Output(
            detections=detections,
            num_detections=count,
            video=Path(out_video_path),
        )
