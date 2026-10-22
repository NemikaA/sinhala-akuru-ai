# sinhala-akuru-ai

A pipeline and web tool for rendering accurate, stylized Sinhala typography onto images.

---

## Overview

Image generation models (Stable Diffusion, Midjourney, FLUX) consistently fail at rendering South Asian scripts like Sinhala. They misplace vowel signs (*pilli*), detach the *kombuwa* (ෙ), and scramble consonant conjuncts (*rakaransaya*, *bandi akuru*) into broken squiggles.

**sinhala-akuru-ai** solves this by separating text shaping from image synthesis:

1. **Deterministic font shaping:** Uses native font shaping engines (HarfBuzz / Windows CTL via `Nirmala UI` and `Noto Sans Sinhala`) to produce mathematically exact glyph masks. Ligatures, diacritics, and left-reordered vowels are placed correctly before any image processing touches them.
2. **Intact image inpainting:** Blends stylized Sinhala text onto an existing photo (storefront sign, wooden board, street wall, mug, t-shirt) while keeping every other pixel of the original photo untouched.
3. **Diffusion conditioning pipeline:** Includes training scripts and synthetic dataset generators for fine-tuning ControlNet adapters with region-weighted loss.

---

## Web Studio

The easiest way to use the project is through the built-in browser UI.

```bash
cd ml-engine
py -m uvicorn server.inference_server:app --port 8000
```

Open **[http://localhost:8000](http://localhost:8000)** in your browser to:
- Upload any image (or test with the built-in storefront sample).
- Type any Sinhala Unicode text (e.g. `ආයුබෝවන්`, `ශ්‍රී ලංකා`, `තේ කඩේ`).
- Select a visual style and download the resulting high-res image.

---

## Visual Styles

| Style | Description |
| :--- | :--- |
| `neon_glow` | Multi-pass glow and bloom with a bright core for cyberpunk or night signs |
| `wood_carved` | Engraved wood sign with chiseled shadows and grain highlights |
| `metallic_gold` | 3D beveled gold/brass finish with upper-left specular highlights |
| `painted_sign` | Acrylic/oil paint texture blended onto wall or board surfaces |
| `chalk` | Soft chalk texture for restaurant and cafe blackboards |
| `clean_white` / `clean_black` | Crisp solid vector text |
| `color_hex` | Custom hex color code (e.g. `#00FFFF`, `#FFB300`) |

---

## Project Structure

```text
sinhala-akuru-ai/
├── app-client/                  # Java client module (native font shaping)
│   ├── build.gradle
│   └── src/main/java/com/sinhalagen/
│       ├── MainApp.java         # CLI test entrypoint
│       ├── glyph/
│       │   └── SinhalaGlyphRenderer.java  # HarfBuzz/Uniscribe glyph rasterizer
│       └── client/
│           └── InferenceClient.java       # HTTP client for Python server
├── ml-engine/                   # Python server, training, and data pipeline
│   ├── requirements.txt
│   ├── server/
│   │   └── inference_server.py  # FastAPI app + browser Studio UI
│   ├── data_pipeline/
│   │   ├── generate_synthetic_data.py # Synthetic paired dataset generator
│   │   └── fonts/               # Bundled Unicode fonts
│   └── training/
│       └── train_glyph_lora.py  # ControlNet training with region-weighted loss
├── build.gradle                 # Root Gradle configuration
├── settings.gradle
└── README.md
```

---

## Setup & Usage

### Requirements
- **Python 3.10+**
- **Java 17+ or 21+** (for the Java client module)
- Windows includes `Nirmala UI` out of the box. On Linux, install `fonts-noto-core`.

### 1. Python Server & Studio

Install requirements:
```bash
cd ml-engine
pip install -r requirements.txt
```

Launch the server:
```bash
py -m uvicorn server.inference_server:app --host 127.0.0.1 --port 8000
```

- Studio UI: `http://localhost:8000`
- API documentation (Swagger): `http://localhost:8000/docs`
- Health check: `http://localhost:8000/health`

### 2. Standalone Java Glyph Engine

You can verify the font shaping engine directly without running the Python server:

```powershell
javac -d bin app-client/src/main/java/com/sinhalagen/glyph/SinhalaGlyphRenderer.java
java -cp bin com.sinhalagen.glyph.SinhalaGlyphRenderer "ආයුබෝවන් ශ්‍රී ලංකා"
```

This outputs clean binary glyph masks to `output/sinhala_glyph_mask.png`.

### 3. Synthetic Dataset Generation

To generate paired training samples (glyph masks, bounding boxes, and stylized scene composites):

```bash
cd ml-engine
py data_pipeline/generate_synthetic_data.py --samples 1000 --out_dir dataset
```

Outputs:
- `dataset/glyphs/`: 512x512 black-and-white glyph condition maps
- `dataset/masks/`: Text bounding box masks
- `dataset/targets/`: Synthetic photo composites
- `dataset/metadata.jsonl`: Prompt and bbox metadata

### 4. Training ControlNet (Optional)

To train a ControlNet adapter on your generated dataset:

```bash
cd ml-engine
py training/train_glyph_lora.py --data_dir dataset --epochs 5 --batch_size 1
```

Weights will be saved to `checkpoints/sinhala-glyph-controlnet`.

---

## API Reference

### `POST /generate`

Accepts JSON payloads. Supports raw text or base64 inputs:

```json
{
  "sinhala_text": "ආයුබෝවන් ශ්‍රී ලංකා",
  "style": "neon_glow",
  "color_hex": "#00FFFF",
  "base_image_base64": "<optional-base64-encoded-image>"
}
```

**Response:**
```json
{
  "status": "success",
  "style_applied": "neon_glow",
  "generated_image_base64": "<base64-output-png>"
}
```

---

## License

This project is licensed under the [MIT License](LICENSE).
