# sinhala-akuru-ai

Render accurate, stylized Sinhala text onto any image — without it turning into broken squiggles.

---

## Why this exists

Every popular image model (Stable Diffusion, FLUX, Midjourney) fails badly at Sinhala. It's not a minor rendering glitch — vowel signs (*pilli*) end up detached, the *kombuwa* gets swapped to the wrong side, and consonant conjuncts (*rakaransaya*, *bandi akuru*) come out as random noise. The models simply weren't trained on enough Sinhala data to ever get this right on their own.

The fix here isn't to hope a bigger model learns Sinhala better. It's to take text rendering out of the model's hands entirely:

1. **Font shaping handles the text.** HarfBuzz / Windows CTL (`Nirmala UI`, `Noto Sans Sinhala`) produce pixel-exact glyph masks — ligatures, diacritics, and left-reordered vowels placed correctly before diffusion touches anything.
2. **Inpainting handles the image.** The styled Sinhala text is blended onto your photo (storefront, sign, t-shirt, mug, wall) while keeping every other pixel of the original untouched.
3. **Training pipeline included.** Scripts for generating synthetic paired datasets and fine-tuning a ControlNet adapter with region-weighted loss, if you want to push further.

---

## Web Studio

The quickest way in. No API calls needed.

```bash
cd ml-engine
py -m uvicorn server.inference_server:app --port 8000
```

Open **[http://localhost:8000](http://localhost:8000)**, upload an image, type your Sinhala text, pick a style, download the result.

---

## Styles

| Style | What it looks like |
| :--- | :--- |
| `neon_glow` | Multi-pass bloom + bright core — good for night scenes and signs |
| `wood_carved` | Engraved wood look with chiseled shadows and grain |
| `metallic_gold` | Beveled gold/brass with specular highlight |
| `painted_sign` | Paint texture blended onto wall or board |
| `chalk` | Soft chalk on dark background — cafe/restaurant boards |
| `clean_white` / `clean_black` | Crisp solid text, no effects |
| `color_hex` | Any hex color you want (`#00FFFF`, `#FFB300`, etc.) |

---

## Project Layout

```
sinhala-akuru-ai/
├── app-client/                  # Java module — native font shaping
│   └── src/main/java/com/sinhalagen/
│       ├── glyph/
│       │   └── SinhalaGlyphRenderer.java   # HarfBuzz/Uniscribe rasterizer
│       └── client/
│           └── InferenceClient.java        # HTTP client for the Python server
└── ml-engine/                   # Python server, training, data pipeline
    ├── server/
    │   └── inference_server.py  # FastAPI app + browser Studio
    ├── data_pipeline/
    │   ├── generate_synthetic_data.py
    │   └── fonts/
    └── training/
        └── train_glyph_lora.py  # ControlNet training with region-weighted loss
```

---

## Setup

**Requirements:** Python 3.10+, Java 17+. Windows already ships with `Nirmala UI`. On Linux, install `fonts-noto-core`.

### Python server

```bash
cd ml-engine
pip install -r requirements.txt
py -m uvicorn server.inference_server:app --host 127.0.0.1 --port 8000
```

- Studio UI → `http://localhost:8000`
- Swagger docs → `http://localhost:8000/docs`
- Health check → `http://localhost:8000/health`

### Java glyph engine (standalone)

Run this to verify the font shaper works without starting the Python server:

```powershell
javac -d bin app-client/src/main/java/com/sinhalagen/glyph/SinhalaGlyphRenderer.java
java -cp bin com.sinhalagen.glyph.SinhalaGlyphRenderer "ජාතික රූපවාහිනී"
```

Outputs a binary glyph mask to `output/sinhala_glyph_mask.png`.

### Generate training data

```bash
cd ml-engine
py data_pipeline/generate_synthetic_data.py --samples 1000 --out_dir dataset
```

Produces:
- `dataset/glyphs/` — 512×512 glyph condition maps
- `dataset/masks/` — bounding box masks
- `dataset/targets/` — synthetic scene composites
- `dataset/metadata.jsonl` — prompt + bbox records

### Train ControlNet (optional)

```bash
cd ml-engine
py training/train_glyph_lora.py --data_dir dataset --epochs 5 --batch_size 1
```

Saves weights to `checkpoints/sinhala-glyph-controlnet`.

---

## API

### `POST /generate`

```json
{
  "sinhala_text": "ජාතික රූපවාහිනී",
  "style": "neon_glow",
  "color_hex": "#00FFFF",
  "base_image_base64": "<optional>"
}
```

```json
{
  "status": "success",
  "style_applied": "neon_glow",
  "generated_image_base64": "<base64-png>"
}
```

---

## License

[MIT](LICENSE)
