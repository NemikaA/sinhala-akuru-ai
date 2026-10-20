# Sinhala Akuru AI (සිංහල අකුරු AI) 🇱🇰✨

**Sinhala Akuru AI** is an end-to-end framework and pipeline for generating photo-realistic images with accurate, legible **Sinhala typography (සිංහල අක්ෂර)** using a **Glyph-Conditioned ControlNet + LoRA** architecture with **Region-Weighted Loss** and deterministic **HarfBuzz text shaping**.

---

## 🎯 Problem Statement

Standard text-to-image diffusion models (Stable Diffusion, FLUX, Midjourney, DALL-E) consistently fail when generating Sinhala script due to two fundamental architectural hurdles:

1. **Tokenizer Fragmentation:** Byte-Pair Encoding (BPE) tokenizers split Sinhala Unicode code points (`U+0D80` to `U+0DFF`) into fragmented byte sequences without semantic typographic understanding.
2. **Complex Text Layout (CTL) Blindness:** Sinhala is an abugida where visual glyph ordering differs from logical Unicode memory order:
   - **Kombuwa (ෙ):** Stored after the consonant in memory, but must be rendered *before* it visually (e.g. `ක` + `ෙ` $\rightarrow$ `කෙ`).
   - **Diacritics & Pillam:** Vowel signs (Ispilla `ි/ී`, Papilla `ු/ූ`, Aeda `ැ/ෑ`, Hal-lakuna `්`) attach dynamically above, below, or beside glyphs.
   - **Ligatures & Conjuncts (බැඳි අකුරු):** Conjuncts such as *Rakaransaya* (e.g. `ක්‍ර`) and *Yansaya* (e.g. `ද්‍ය`) require Zero-Width Joiner (`U+200D`) ligature substitution.

Without deterministic shaping, generative models hallucinate broken squiggles.

---

## 🧠 Solution Architecture

Instead of retraining a billion-parameter foundation model from scratch, **Sinhala Akuru AI** uses a two-tier hybrid architecture:

```
[User Sinhala Text] + [Target Box]
          │
          ▼
┌─────────────────────────────────────────┐
│ Java Client (Native HarfBuzz Engine)   │
│  - java.awt.font.TextLayout             │
│  - Deterministic Unicode Ligature Shape │
└──────────────────┬──────────────────────┘
                   │ Base64 Payloads
       ┌───────────┴───────────┐
       ▼                       ▼
┌──────────────┐       ┌──────────────┐
│ Glyph Mask   │       │ Region Mask  │
│ (White/Black)│       │ (Latent BBox)│
└──────┬───────┘       └──────┬───────┘
       │                      │
       ▼                      ▼
┌─────────────────────────────────────────┐
│ Python ML Engine (PyTorch / FastAPI)    │
│  - ControlNet Spatial Conditioning      │
│  - Base Model: Stable Diffusion v1.5    │
│  - Region-Weighted Loss (λ = 5.0)       │
└──────────────────┬──────────────────────┘
                   │
                   ▼
       [Generated Scene Image]
```

### 1. Deterministic HarfBuzz Text Shaping
Java's `java.awt.font.TextLayout` leverages the native **HarfBuzz** shaping engine, guaranteeing 100% accurate rendering of vowel diacritics and ligatures without guessing.

### 2. Region-Weighted Diffusion Loss
Because Sinhala diacritics (like *Al-lakuna* `්` or *Ispilla* `ි`) occupy only a small fraction of pixels in a 512×512 image, standard mean-squared error loss tends to smooth them away. We supervise the network with an amplified loss inside the text bounding box $M$:

$$\mathcal{L}_{\text{total}} = \mathbb{E}_{z_0, t, c_{\text{prompt}}, c_{\text{glyph}}, \epsilon} \left[ \left\| (1 + \lambda \cdot M) \odot (\epsilon - \epsilon_\theta(z_t, t, c_{\text{prompt}}, c_{\text{glyph}})) \right\|_2^2 \right]$$

where $\lambda = 5.0$ applies $5\times$ sharper supervision inside the Sinhala typographic region.

---

## 📂 Repository Structure

This repository is organized as a multi-module project orchestrated with Gradle:

```text
sinhala-akuru-ai/
├── build.gradle                          # Root Gradle orchestration (Java + Python tasks)
├── settings.gradle                       # Gradle project settings
├── app-client/                           # Java 21 Client Module
│   ├── build.gradle                      # Java application dependencies & plugins
│   └── src/
│       ├── main/java/com/sinhalagen/
│       │   ├── MainApp.java              # Client entrypoint & verification
│       │   ├── glyph/
│       │   │   └── SinhalaGlyphRenderer.java  # HarfBuzz TextLayout mask generator
│       │   └── client/
│       │       └── InferenceClient.java       # HTTP REST client to Python engine
│       └── test/java/com/sinhalagen/
│           └── glyph/
│               └── SinhalaGlyphRendererTest.java  # Unit tests for complex conjuncts
└── ml-engine/                            # Python PyTorch & FastAPI Engine
    ├── requirements.txt                  # Python dependencies
    ├── data_pipeline/
    │   ├── fonts/                        # Unicode Sinhala TrueType fonts
    │   │   └── download_font.py          # Font downloader utility
    │   └── generate_synthetic_data.py    # Paired dataset generator (Glyph, Target, Mask)
    ├── training/
    │   └── train_glyph_lora.py           # ControlNet training with region-weighted loss
    └── server/
        └── inference_server.py           # FastAPI service exposing /generate endpoint
```

---

## 🎨 Intact Image Styling & Inpainting

You can take **any existing image** (e.g. a storefront, signboard, blank mug, street wall, t-shirt, or poster), and the model will inpaint the stylized Sinhala text onto it while keeping the rest of the original photo **100% intact**:

### Supported Graphical Styles
* `neon_glow`: Cyberpunk / night illuminated sign with multi-radius bloom.
* `metallic_gold`: 3D embossed gold/brass with bevel highlights and drop shadows.
* `wood_carved`: Deep-carved wooden sign with shadow and grain blending.
* `painted_sign`: Acrylic / oil street paint with subtle edge texture.
* `chalk`: Dusty white / colored chalk texture for cafe boards.
* `clean_white` / `clean_black`: Crisp modern solid typography.
* `color_hex`: Custom hex color code (e.g. `#00FFFF`, `#FF9900`).

---

## 🚀 Quickstart & Workflow

### Prerequisites
- **Java Development Kit (JDK 21 or later)**
- **Python 3.10+** (with PyTorch and CUDA support if training/inferring with GPU)
- **Gradle 8+** (or system wrapper)

---

### Step 1: Set Up Python ML Environment
Install required PyTorch, Diffusers, and FastAPI libraries:
```bash
./gradlew setupMlEnv
# or manually:
cd ml-engine && pip install -r requirements.txt
```

To fetch the reference open-source font (`NotoSansSinhala-Bold.ttf`):
```bash
python ml-engine/data_pipeline/fonts/download_font.py
```

---

### Step 2: Generate Paired Synthetic Dataset
Generate synthetic pairs consisting of:
1. `glyphs/` (White-on-black glyph condition map)
2. `masks/` (Bounding box region mask)
3. `targets/` (Realistic background with composite shaded Sinhala text)
4. `metadata.jsonl`

```bash
./gradlew generateDataset
# or with custom sample count:
python ml-engine/data_pipeline/generate_synthetic_data.py --samples 5000 --out_dir dataset
```

---

### Step 3: Train ControlNet Model
Train the Sinhala glyph conditioning adapter using region-weighted loss:
```bash
./gradlew trainModel
# or directly:
python ml-engine/training/train_glyph_lora.py --epochs 5 --batch_size 1 --region_weight_lambda 5.0
```
Trained weights will be saved to `checkpoints/sinhala-glyph-controlnet`.

---

### Step 4: Run the Inference Server
Launch the FastAPI server on `http://127.0.0.1:8000`:
```bash
./gradlew startInferenceServer
```

Interactive API documentation will be accessible at:
- Swagger UI: `http://127.0.0.1:8000/docs`
- Health check: `http://127.0.0.1:8000/health`

---

### Step 5: Test & Run via Java Client
Generate test glyph condition masks directly using Java HarfBuzz:
```bash
./gradlew run
```

Send an end-to-end inference request to the server:
```bash
./gradlew run --args="--infer"
```

To run unit tests verifying complex Sinhala conjuncts:
```bash
./gradlew test
```

---

## 🗺️ Roadmap & Milestones

| Milestone | Deliverable | Validation Metric |
| :--- | :--- | :--- |
| **Step 1: HarfBuzz Verification** | `SinhalaGlyphRenderer` with complex ligatures (ශ්‍රී, ක්‍ර, ෞ, ෙ). | Zero detached kombuwa or broken Unicode rings on mask PNG. |
| **Step 2: Synthetic Dataset** | 10,000+ pairs using 5–8 Unicode Sinhala fonts and background scenes. | Visual check of perspective-warped Sinhala signage blended onto photos. |
| **Step 3: ControlNet + Region Loss** | Training at 512×512 using fp16 and $\lambda = 5.0$ box weighting. | Character Error Rate (CER) measured with Tesseract OCR (`sin.traineddata`). |
| **Step 4: Interactive Studio** | JavaFX canvas allowing user to draw a text bounding box and enter prompt. | End-to-end latency < 6s per image. |

---

## 📄 License
This project is licensed under the [MIT License](LICENSE).
