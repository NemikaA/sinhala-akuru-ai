# Sinhala Akuru AI (සිංහල අකුරු AI) 🇱🇰✨

Fixing and enhancing **Sinhala visual text and glyph generation** in modern AI image generation models (Diffusion models, FLUX, SDXL, etc.).

---

## 🎯 Problem Statement

Most modern text-to-image models (Stable Diffusion, FLUX, Midjourney, DALL-E) struggle significantly with complex Brahmic and Indic scripts—particularly **Sinhala (සිංහල)**. 

Common issues include:
- **Glyph Distortion:** Incorrect curves, loops, and disconnected strokes.
- **Diacritics & Modifiers (පිල්ලම් / Pillam):** Misplaced or missing Kombuwa (ෙ), Aeda (ැ/ෑ), Al-lakuna (්), Ispilla (ි/ී), and Papilla (ු/ූ).
- **Ligatures & Consonant Conjuncts (බැඳි අකුරු / Bandi Akuru):** Failure to render complex conjuncts properly.
- **Vowel Signs Ordering:** Spatial positioning of left-side vowel signs (like Kombuwa) placed after the consonant.

---

## 🚀 Objectives

1. **Sinhala Glyph Conditioning Pipeline:** Provide glyph-aware conditioning mechanisms (e.g., ControlNet, Glyph-By-T5, AnyText adaptation) for Sinhala typography.
2. **Dataset & Benchmark:** Build a curated dataset of Sinhala font renderings, synthetic signs, banners, and typographic compositions with exact bounding box & character annotations.
3. **Fine-Tuning / LoRA Weights:** Train LoRA adapters and fine-tuned checkpoints specifically for Sinhala visual text synthesis.
4. **Interactive Studio / API:** A lightweight UI or pipeline interface for generating images containing verified, legible Sinhala text.

---

## 📂 Project Structure

```text
sinhala-akuru-ai/
├── data/              # Dataset collection, glyph rendering, annotations
│   ├── fonts/         # Open-source Sinhala Unicode fonts
│   └── raw/           # Raw imagery & text pairings
├── src/               # Core source code
│   ├── rendering/     # Glyph / PIL / HarfBuzz text rendering utilities
│   ├── dataset/       # Data loaders and synthetic text generators
│   ├── models/        # Model definitions, ControlNet / LoRA pipelines
│   └── utils/         # Helper functions and text validation
├── notebooks/         # Exploratory data analysis and experiments
├── configs/           # Training and inference configurations
├── requirements.txt   # Python dependencies
└── README.md
```

---

## 🛠️ Quickstart (Coming Soon)

### Installation
```bash
git clone https://github.com/NemikaA/sinhala-akuru-ai.git
cd sinhala-akuru-ai
pip install -r requirements.txt
```

---

## 📄 License
This project is licensed under the [MIT License](LICENSE).
