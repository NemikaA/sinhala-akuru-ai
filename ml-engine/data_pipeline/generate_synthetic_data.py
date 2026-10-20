import os
import json
import random
import argparse
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

# Diverse Sinhala vocabulary covering everyday signage, names, and complex ligatures
SINHALA_PHRASES = [
    "ආයුබෝවන්", "ශ්‍රී ලංකා", "මහනුවර", "කොළඹ", "තේ කඩේ",
    "සිංහල අලුත් අවුරුද්ද", "ගාලු මුවදොර", "පොත් හල", "සුභ උපන්දිනයක්", "රස කැවිලි",
    "හෝටලය සහ බේකරිය", "පැරණි බෙහෙත් ශාලාව", "බස් නැවතුම", "ස්වර්ණාභරණ",
    "පලතුරු සහ එළවළු", "මල් ශාලාව", "සීගිරිය", "නුවරඑළිය", "යාපනය", "මිහින්තලේ"
]

SCENE_TEMPLATES = [
    ("a vintage wooden signboard hanging outside a traditional shop with text", (210, 180, 140)),
    ("a glowing cyberpunk neon sign at night in a rainy city street displaying text", (0, 255, 220)),
    ("a minimalist ceramic coffee mug on a wooden cafe table printed with text", (40, 40, 40)),
    ("a painted street wall mural in Colombo featuring bold text", (255, 215, 0)),
    ("a rustic metal street signboard weathered by sunlight with text", (220, 220, 230)),
    ("a cozy cafe chalkboard menu board displaying chalk text", (240, 240, 240)),
    ("an authentic Sri Lankan bus destination board with illuminated text", (255, 50, 50)),
]

def get_sinhala_font(font_path: str, font_size: int):
    """Loads specified TTF font or falls back to system Sinhala fonts."""
    candidates = [
        font_path,
        "ml-engine/data_pipeline/fonts/NotoSansSinhala-Bold.ttf",
        "data_pipeline/fonts/NotoSansSinhala-Bold.ttf",
        "C:/Windows/Fonts/iskpota.ttf",      # Windows Iskoola Pota
        "C:/Windows/Fonts/iskpotab.ttf",     # Windows Iskoola Pota Bold
        "C:/Windows/Fonts/Nirmala.ttf",      # Windows Nirmala UI
        "/usr/share/fonts/truetype/noto/NotoSansSinhala-Bold.ttf",
    ]
    for c in candidates:
        if c and os.path.exists(c):
            try:
                return ImageFont.truetype(c, font_size)
            except Exception:
                pass
    try:
        return ImageFont.load_default()
    except Exception:
        return None

def create_sample(idx: int, font: ImageFont.ImageFont, out_dir: str):
    img_size = 512
    phrase = random.choice(SINHALA_PHRASES)
    prompt_desc, text_color = random.choice(SCENE_TEMPLATES)

    # 1. Create Glyph Condition Map (strictly white glyph on black background)
    glyph_img = Image.new("RGB", (img_size, img_size), (0, 0, 0))
    mask_img = Image.new("L", (img_size, img_size), 0)

    draw_glyph = ImageDraw.Draw(glyph_img)
    try:
        bbox = draw_glyph.textbbox((0, 0), phrase, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    except Exception:
        tw, th = 260, 50

    x = random.randint(40, max(41, img_size - tw - 40))
    y = random.randint(100, max(101, img_size - th - 100))

    draw_glyph.text((x, y), phrase, font=font, fill=(255, 255, 255))

    # 2. Create Region Bounding Box Mask (for region-weighted diffusion loss)
    draw_mask = ImageDraw.Draw(mask_img)
    pad = 14
    draw_mask.rectangle([x - pad, y - pad, x + tw + pad, y + th + pad], fill=255)

    # 3. Synthesize Realistic Target Image (textured background + shaded/blended text)
    bg_noise = np.random.randint(30, 95, (img_size, img_size, 3), dtype=np.uint8)
    bg_img = Image.fromarray(cv2.GaussianBlur(bg_noise, (25, 25), 0))

    text_layer = Image.new("RGBA", (img_size, img_size), (0, 0, 0, 0))
    draw_target = ImageDraw.Draw(text_layer)

    # Drop shadow
    draw_target.text((x + 3, y + 3), phrase, font=font, fill=(0, 0, 0, 190))
    # Stylized main Sinhala text
    draw_target.text((x, y), phrase, font=font, fill=(*text_color, 245))
    text_layer = text_layer.filter(ImageFilter.GaussianBlur(radius=0.5))

    target_img = Image.alpha_composite(bg_img.convert("RGBA"), text_layer).convert("RGB")

    # Save relative files
    glyph_rel = f"glyphs/{idx:05d}.png"
    mask_rel = f"masks/{idx:05d}.png"
    target_rel = f"targets/{idx:05d}.png"

    glyph_img.save(os.path.join(out_dir, glyph_rel))
    mask_img.save(os.path.join(out_dir, mask_rel))
    target_img.save(os.path.join(out_dir, target_rel))

    return {
        "id": idx,
        "prompt": f"{prompt_desc}: \"{phrase}\"",
        "sinhala_text": phrase,
        "glyph_path": glyph_rel,
        "mask_path": mask_rel,
        "target_path": target_rel,
        "bbox": [x - pad, y - pad, x + tw + pad, y + th + pad]
    }

def main():
    parser = argparse.ArgumentParser(description="Synthetic Sinhala glyph-scene pair generator")
    parser.add_argument("--samples", type=int, default=1000, help="Number of synthetic pairs to generate")
    parser.add_argument("--out_dir", type=str, default="dataset", help="Output directory for generated dataset")
    parser.add_argument("--font", type=str, default="ml-engine/data_pipeline/fonts/NotoSansSinhala-Bold.ttf", help="Font file path")
    args = parser.parse_args()

    for sub in ["glyphs", "masks", "targets"]:
        os.makedirs(os.path.join(args.out_dir, sub), exist_ok=True)

    font = get_sinhala_font(args.font, font_size=52)
    print(f"Generating {args.samples} synthetic Sinhala glyph-scene pairs into '{args.out_dir}'...")

    metadata = []
    for i in range(args.samples):
        sample = create_sample(i, font, args.out_dir)
        metadata.append(sample)
        if (i + 1) % 100 == 0 or (i + 1) == args.samples:
            print(f"  Progress: {i + 1}/{args.samples} generated")

    meta_file = os.path.join(args.out_dir, "metadata.jsonl")
    with open(meta_file, "w", encoding="utf-8") as f:
        for entry in metadata:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"✔ Completed. Saved metadata index to: {meta_file}")

if __name__ == "__main__":
    main()
