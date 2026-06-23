# LocateAnything-3B — Visual Grounding (imagem + vídeo)

Wrapper Cog do [`nvidia/LocateAnything-3B`](https://huggingface.co/nvidia/LocateAnything-3B) — um modelo de visão-linguagem (MoonViT + Qwen2.5-3B + **Parallel Box Decoding**) para **localização de objetos** a partir de linguagem natural.

Dada uma **imagem ou vídeo** + uma instrução em texto, devolve rótulos com **bounding boxes** e **points**, além da **mídia anotada** (PNG para imagem, MP4 H.264 para vídeo).

> Forte em **detecção densa em cena cheia** (contar centenas/milhares de objetos numa única imagem), referring grounding, GUI grounding, OCR/layout e pointing.

---

## 🟢 Em palavras simples (comece por aqui)

Pensa nele como um **"caça-objetos por comando de texto"**:

> Você mostra uma **foto** (ou um **vídeo**) e escreve, em inglês, **o que procurar** — tipo *"Detect all the cars"* (detecte todos os carros). Ele devolve a mesma imagem com **quadradinhos verdes** em volta de cada objeto encontrado, e te diz **quantos** achou.

Serve pra coisas como:
- **Contar** — quantas cabeças de gado, ovos, pessoas, peças tem nesta foto?
- **Achar** — onde está a placa de saída / o carro vermelho / o produto X?
- **Marcar em vídeo** — desenhar caixas nos objetos ao longo de um vídeo.

Você **não precisa** mexer em quase nada: joga a foto, escreve o que quer em inglês, e pronto. Os "controles avançados" abaixo só servem pra ajustar casos específicos.

---

## Dois modelos (escolha o hardware)

| Modelo | Hardware | Quando usar |
|---|---|---|
| `csviverdeia/locateanything-3b-h100` | **H100** | **Padrão para tudo.** ~2,4× mais rápido; vídeo ~40% mais barato/job; imagem custa praticamente igual ao L40S porém mais rápida. |
| `csviverdeia/locateanything-3b` | L40S | Fallback de disponibilidade (H100 é GPU disputada). Mesmo código. |

**Recomendação:** use o **H100** como principal. Mantenha o L40S como reserva — você só paga quando roda.

---

## Inputs

| Input | Tipo | Default | Faixa | Descrição |
|---|---|---|---|---|
| `image` | file | — | — | Imagem RGB. Use `image` **ou** `video` (o modelo detecta o tipo do arquivo sozinho). |
| `video` | file | — | — | Vídeo. É amostrado em frames e detectado quadro-a-quadro. |
| `prompt` | str | "Detect all the main objects…" | — | O que localizar. Ver **Dicas de prompt**. |
| `generation_mode` | enum | `hybrid` | hybrid / ar / mtp | Modo de decodificação. `hybrid` equilibra velocidade e precisão. |
| `tracker` | enum | `none` | none / sort / reid | **[vídeo]** `none` = só caixas, sem ID (estilo NVIDIA). `sort` = IDs por movimento (Kalman+Hungarian). `reid` = SORT + aparência (ResNet18). |
| `detect_fps` | float | 3.0 | 0.5–10 | **[vídeo]** Detecções por segundo. Mais alto = acompanha melhor o movimento, mais lento/caro. |
| `max_detect_frames` | int | 24 | 1–120 | **[vídeo]** Teto de frames detectados (distribuídos no vídeo) p/ limitar custo. |
| `max_new_tokens` | int | 2048 | 64–8192 | Limite de tokens de saída. Suba para cenas MUITO densas (centenas de boxes). |
| `temperature` | float | 0.2 | 0–2 | `0` = determinístico (recomendado p/ contagem; reduz falso-positivo). |
| `top_p` | float | 0.9 | 0–1 | Amostragem nucleus. |
| `repetition_penalty` | float | 1.1 | 1–2 | Evita repetição. |

### 🧩 Entendendo os controles (em português claro)

- **`image` / `video`** — o arquivo que você sobe. Tanto faz em qual campo: se for vídeo, ele entende sozinho. Use **um** dos dois.
- **`prompt`** — a frase (em inglês) dizendo o que procurar. É o controle **mais importante**. Ex.: `"Detect all the people"` (ache todas as pessoas). Seja específico.
- **`temperature`** — o "quão criativo" ele é. **Pra contar, deixe `0`** (mais preciso, não inventa). Valores altos = mais chute.
- **`max_new_tokens`** — quanto ele "pode falar". Cada objeto gasta um pouco. Se a foto tem **muitos** objetos (centenas) e a contagem parece cortada, **aumente** (ex.: 8192).
- **`detect_fps`** *(só vídeo)* — **quantas vezes por segundo** ele olha o vídeo. Objeto **rápido** (carro, esteira) → número **alto** (5–10), pra acompanhar. Objeto **lento** → 2–3 já basta. Quanto maior, mais lento e mais caro.
- **`max_detect_frames`** *(só vídeo)* — um **limite de segurança** de quantos quadros ele analisa (pra não ficar caro/demorado em vídeo longo). Se quiser mais precisão num vídeo, **aumente**.
- **`tracker`** *(só vídeo)* — como ele lida com a **identidade** dos objetos:
  - **`none`** (padrão) → só desenha as caixas, **sem número**. É o mais limpo e o que a NVIDIA mostra. **Use este na dúvida.**
  - **`sort`** → coloca um **número fixo** (#1, #2…) em cada objeto e tenta manter ao longo do vídeo. Bom pra objetos **diferentes** e movimento normal.
  - **`reid`** → igual ao `sort`, mas também "lembra a aparência" — segura melhor o número quando o objeto some atrás de algo e volta. Mais lento. Bom pra **cruzamentos/oclusão**.
- **`generation_mode` / `top_p` / `repetition_penalty`** — controles finos; **deixe no padrão** a não ser que você saiba o que está fazendo.

---

## 🍳 Receitas — qual configuração usar em cada situação

| Situação | Configuração recomendada |
|---|---|
| **Contar objetos numa foto** (gado, ovos, peças) | `image` + `prompt: "Detect every individual X. Output one tight box per X."` + **`temperature: 0`** + `max_new_tokens: 8192` |
| **Achar UMA coisa específica** numa foto | `image` + `prompt: "Detect the red car"` (descreva bem) |
| **Marcar um ponto** em cada objeto (não caixa) | `image` + `prompt: "Point to each person."` |
| **Vídeo bonito e fluido** (estilo NVIDIA, sem números) | `video` + **`tracker: none`** + `detect_fps: 5` (ou mais) + `max_detect_frames: 60` |
| **Vídeo com objetos numerados** (poucos, distintos) | `video` + `tracker: sort` + `detect_fps: 4` |
| **Vídeo com cruzamento/oclusão** (objetos somem e voltam) | `video` + `tracker: reid` + `detect_fps: 4` |
| **Objetos rápidos** (trânsito, esteira) | suba o **`detect_fps`** (8–10) e o `max_detect_frames` proporcional |
| **Vídeo longo, quer economizar** | `detect_fps: 2` + `max_detect_frames: 24` (menos quadros = mais barato) |
| **A contagem veio "cortada"** (muitos objetos) | aumente **`max_new_tokens`** (8192) |
| **Ele inventou caixas onde não tem nada** | **`temperature: 0`** |

> **Regra de ouro:** pra **contar**, use **foto** (não vídeo) com `temperature: 0`. Pra **mostrar movimento** num vídeo, use `tracker: none`. O resto é ajuste fino.

## Output

```json
{
  "detections": "<JSON: modality, answer, coords, boxes[], points[], ...>",
  "num_detections": 36,
  "image": "https://.../annotated.png",   // só para input de imagem
  "video": "https://.../annotated.mp4"    // só para input de vídeo (H.264)
}
```

- **Imagem:** `boxes`/`points` em **pixels**; `image` traz a foto com as caixas numeradas.
- **Vídeo:** `video` traz o MP4 anotado; `detections` traz o log por frame. No modo `none`, `num_detections` = **pico de objetos simultâneos**; em `sort`/`reid` = **objetos únicos** rastreados.

---

## Casos de uso

| Objetivo | Como |
|---|---|
| **Contar objetos** (ovos, gado, pacotes, peças) | **1 imagem boa** + prompt de detecção densa. É o forte do modelo. |
| **Achar algo específico** | `prompt` = "Detect the red car" / "Locate the exit sign". |
| **Pointing** (1 ponto por objeto) | `prompt` = "Point to each person." → sai em `points`. |
| **Detecção em vídeo** (visualizar) | `video` + `tracker: none` (caixas limpas, sem renumeração). |
| **Rastrear IDs em vídeo** | `tracker: sort` (objetos distintos) ou `reid` (sobrevive a oclusão). |

---

## Exemplos

### Python (imagem)
```python
import replicate
out = replicate.run(
    "csviverdeia/locateanything-3b-h100",
    input={
        "image": open("herd.jpg", "rb"),
        "prompt": "Detect every individual animal. Output one tight box per animal.",
        "temperature": 0,
        "max_new_tokens": 8192,
    },
)
print(out["num_detections"], out["image"])
```

### Python (vídeo, estilo NVIDIA — só caixas)
```python
out = replicate.run(
    "csviverdeia/locateanything-3b-h100",
    input={
        "video": open("street.mp4", "rb"),
        "prompt": "Detect all the cars.",
        "tracker": "none",
        "detect_fps": 5,          # mais alto = mais fluido
        "max_detect_frames": 60,  # suba p/ vídeos longos
    },
)
print(out["video"])
```

### curl (imagem via base64)
```bash
curl -s -X POST https://api.replicate.com/v1/predictions \
  -H "Authorization: Bearer $REPLICATE_API_TOKEN" -H "Content-Type: application/json" \
  -d '{"version":"<VERSION>","input":{"image":"data:image/jpeg;base64,'"$(base64 -w0 img.jpg)"'","prompt":"Detect all cows.","temperature":0}}'
```

---

## Dicas de prompt

- **Detecção:** *"Detect every individual X. Output one tight bounding box per X."*
- **Contagem densa:** seja explícito em "cada/individual" e use `temperature: 0` + `max_new_tokens` alto.
- **Pointing:** *"Point to each X."* (saída vira `points`).
- **Referring:** *"Locate all the instances that match: <descrição>."*
- Objeto ausente → o modelo pode **alucinar 1–2 caixas**; `temperature: 0` reduz.

## Dicas de vídeo

- **Velocidade do objeto** manda no `detect_fps`: objeto rápido (esteira, trânsito) → suba para 5–10. Lento → 2–3 basta.
- `max_detect_frames` precisa ser **alto o bastante** p/ não sufocar o `detect_fps` (senão a detecção fica rara e as caixas "pulam").
- Para o resultado **mais fluido (estilo NVIDIA)**: `tracker: none` + `detect_fps` ≈ fps do vídeo. Custa mais (uma inferência por frame) — por isso o **H100**.
- **Contar throughput em esteira** não é o forte: o modelo é detector por frame, não tracker industrial. Para isso, prefira contagem por imagem ou um pipeline dedicado (YOLO+ByteTrack).

---

## Custo (compute, modelo quente)

| | L40S | H100 |
|---|---|---|
| Imagem | ~$0.0016 | ~$0.0017 |
| Vídeo (por frame detectado) | ~$0.0044 | ~$0.0026 |
| Vídeo 5s (detect_fps 3) | ~$0.08 | ~$0.05 |
| Vídeo 10s todo-frame (300 fr) | ~$1.32 | ~$0.79 |

Custo de vídeo ≈ `frames_detectados × (0.0044 L40S / 0.0026 H100)`, onde `frames = duração × detect_fps` (limitado por `max_detect_frames`). O 1º hit após ocioso tem cold-start (não incluído acima).

---

## Limitações (honestas)

- **Teto de resolução:** o modelo reduz internamente imagens acima de ~**803k px (~896×896)** (`in_token_limit=4096`). Objetos muito pequenos/ao fundo podem sumir mesmo enviando foto grande. (Para esses casos: tiling — não incluído neste wrapper.)
- **Vídeo é detecção por frame**, não tracking nativo. Cenas densas/rápidas (esteira) causam IDs instáveis em `sort`/`reid` — use `none`.
- **Alucinação:** pode inventar 1–2 caixas quando o objeto pedido não existe.
- **Licença:** o modelo base é **NVIDIA License (uso não-comercial)**. Revise antes de produção.

## Crédito

Modelo: [nvidia/LocateAnything-3B](https://huggingface.co/nvidia/LocateAnything-3B). Wrapper Cog por csviverdeia.
