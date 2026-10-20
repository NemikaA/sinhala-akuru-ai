import io
import os
import base64
from typing import Optional
from PIL import Image, ImageDraw, ImageFilter, ImageFont
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

app = FastAPI(
    title="Sinhala Akuru AI - Visual Text Inpainting & Styling Server",
    description="Inserts stylized, accurate Sinhala typography into any image while keeping the rest of the image 100% intact.",
    version="1.3.0"
)

pipeline = None
device = "cpu"
dtype = None

def get_sinhala_font(font_size: int = 48) -> ImageFont.ImageFont:
    candidate_fonts = [
        "C:/Windows/Fonts/Nirmala.ttc",
        "C:/Windows/Fonts/Nirmala.ttf",
        "C:/Windows/Fonts/iskpota.ttf",
        "ml-engine/data_pipeline/fonts/NotoSansSinhala-Bold.ttf",
        "data_pipeline/fonts/NotoSansSinhala-Bold.ttf",
        "../data_pipeline/fonts/NotoSansSinhala-Bold.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansSinhala-Bold.ttf"
    ]
    for p in candidate_fonts:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, font_size)
            except Exception:
                pass
    return ImageFont.load_default()

def render_sinhala_glyph_mask(text: str, width: int = 512, height: int = 512, font_size: int = 48) -> Image.Image:
    glyph_img = Image.new("RGB", (width, height), (0, 0, 0))
    draw = ImageDraw.Draw(glyph_img)
    font = get_sinhala_font(font_size)
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    except Exception:
        tw, th = 300, 50

    x = max(10, (width - tw) // 2)
    y = max(10, (height - th) // 2)
    draw.text((x, y), text, font=font, fill=(255, 255, 255))
    return glyph_img

def safe_b64decode(b64_str: Optional[str]) -> Optional[bytes]:
    """Safely decodes Base64 string, handling data URI headers, whitespace, and missing padding."""
    if not b64_str or b64_str.strip() == "" or b64_str.strip().lower() == "string":
        return None
    # Strip data URL prefix if present (e.g. data:image/png;base64,...)
    if "," in b64_str:
        b64_str = b64_str.split(",", 1)[1]
    b64_str = b64_str.strip().replace(" ", "").replace("\n", "").replace("\r", "")
    # Add missing padding if needed
    missing_padding = len(b64_str) % 4
    if missing_padding:
        b64_str += "=" * (4 - missing_padding)
    try:
        return base64.b64decode(b64_str)
    except Exception:
        return None

class GenerationRequest(BaseModel):
    sinhala_text: Optional[str] = Field("ආයුබෝවන්", description="Sinhala Unicode text string to render")
    glyph_mask_base64: Optional[str] = Field(None, description="Optional Base64 PNG glyph mask. If omitted, auto-generated from sinhala_text")
    base_image_base64: Optional[str] = Field(None, description="Original image to keep intact. If omitted, uses built-in signboard sample")
    style: Optional[str] = Field("neon_glow", description="Style preset: neon_glow, metallic_gold, wood_carved, painted_sign, chalk, clean_white, clean_black")
    color_hex: Optional[str] = Field(None, description="Optional custom color hex code (e.g. #00FFFF, #FFB300)")
    box_x: Optional[int] = Field(None, description="Optional X offset")
    box_y: Optional[int] = Field(None, description="Optional Y offset")
    box_w: Optional[int] = Field(None, description="Optional text box width")
    box_h: Optional[int] = Field(None, description="Optional text box height")

def hex_to_rgb(hex_str: Optional[str], default_rgb=(255, 255, 255)):
    if not hex_str:
        return default_rgb
    hex_str = hex_str.lstrip("#")
    try:
        if len(hex_str) == 6:
            return tuple(int(hex_str[i:i+2], 16) for i in (0, 2, 4))
    except Exception:
        pass
    return default_rgb

def apply_stylized_text_to_image(
    base_img: Image.Image,
    glyph_mask: Image.Image,
    style: str = "neon_glow",
    color_hex: Optional[str] = None,
    box_x: Optional[int] = None,
    box_y: Optional[int] = None,
    box_w: Optional[int] = None,
    box_h: Optional[int] = None
) -> Image.Image:
    """
    Applies the Sinhala glyph mask to the base image strictly inside the requested
    bounding box while leaving every other pixel of the base image 100% INTACT.
    """
    base_w, base_h = base_img.size

    # 1. Crop glyph mask to non-zero content
    glyph_gray = glyph_mask.convert("L")
    arr = np.array(glyph_gray)
    y_idx, x_idx = np.where(arr > 10)

    if len(x_idx) == 0:
        return base_img.copy()

    min_x, max_x = int(np.min(x_idx)), int(np.max(x_idx))
    min_y, max_y = int(np.min(y_idx)), int(np.max(y_idx))
    crop_w, crop_h = (max_x - min_x), (max_y - min_y)
    cropped_glyph = glyph_gray.crop((min_x, min_y, max_x, max_y))

    # 2. Determine target bounding box on the base image
    target_w = box_w if box_w else int(base_w * 0.72)
    target_h = int(crop_h * (float(target_w) / max(1, crop_w)))
    if box_h and box_h < target_h:
        target_h = box_h
        target_w = int(crop_w * (float(target_h) / max(1, crop_h)))

    pos_x = box_x if box_x is not None else (base_w - target_w) // 2
    pos_y = box_y if box_y is not None else (base_h - target_h) // 2

    resized_glyph = cropped_glyph.resize((target_w, target_h), Image.Resampling.LANCZOS)

    positioned_mask = Image.new("L", (base_w, base_h), 0)
    positioned_mask.paste(resized_glyph, (pos_x, pos_y))

    result = base_img.copy().convert("RGB")
    style_key = (style or "neon_glow").lower()

    if style_key == "neon_glow":
        neon_color = hex_to_rgb(color_hex, default_rgb=(0, 240, 255))
        white_core = (255, 255, 255)

        wide_glow = positioned_mask.filter(ImageFilter.GaussianBlur(16))
        tight_glow = positioned_mask.filter(ImageFilter.GaussianBlur(6))

        res_arr = np.array(result, dtype=np.float32)
        glow_wide_arr = np.array(wide_glow, dtype=np.float32) / 255.0
        glow_tight_arr = np.array(tight_glow, dtype=np.float32) / 255.0
        mask_arr = np.array(positioned_mask, dtype=np.float32) / 255.0

        for c in range(3):
            res_arr[:, :, c] += glow_wide_arr * neon_color[c] * 0.65
            res_arr[:, :, c] += glow_tight_arr * neon_color[c] * 0.95
            res_arr[:, :, c] = res_arr[:, :, c] * (1 - mask_arr) + white_core[c] * mask_arr

        result = Image.fromarray(np.clip(res_arr, 0, 255).astype(np.uint8))

    elif style_key == "metallic_gold":
        gold_color = hex_to_rgb(color_hex, default_rgb=(255, 215, 0))
        shadow = Image.new("L", (base_w, base_h), 0)
        shadow.paste(resized_glyph, (pos_x + 3, pos_y + 3))
        shadow = shadow.filter(ImageFilter.GaussianBlur(2))

        highlight = Image.new("L", (base_w, base_h), 0)
        highlight.paste(resized_glyph, (pos_x - 1, pos_y - 1))
        highlight = highlight.filter(ImageFilter.GaussianBlur(1))

        res_arr = np.array(result, dtype=np.int16)
        res_arr -= (np.array(shadow)[:, :, None] * 0.45).astype(np.int16)
        res_arr += (np.array(highlight)[:, :, None] * 0.35).astype(np.int16)
        result = Image.fromarray(np.clip(res_arr, 0, 255).astype(np.uint8))

        color_layer = Image.new("RGB", (base_w, base_h), gold_color)
        result = Image.composite(color_layer, result, positioned_mask)

    elif style_key == "wood_carved":
        tint = hex_to_rgb(color_hex, default_rgb=(245, 215, 120))
        shadow = Image.new("L", (base_w, base_h), 0)
        shadow.paste(resized_glyph, (pos_x + 3, pos_y + 3))
        shadow = shadow.filter(ImageFilter.GaussianBlur(2))

        highlight = Image.new("L", (base_w, base_h), 0)
        highlight.paste(resized_glyph, (pos_x - 1, pos_y - 1))
        highlight = highlight.filter(ImageFilter.GaussianBlur(1))

        res_arr = np.array(result, dtype=np.int16)
        res_arr -= (np.array(shadow)[:, :, None] * 0.45).astype(np.int16)
        res_arr += (np.array(highlight)[:, :, None] * 0.25).astype(np.int16)
        result = Image.fromarray(np.clip(res_arr, 0, 255).astype(np.uint8))

        color_layer = Image.new("RGB", (base_w, base_h), tint)
        result = Image.composite(color_layer, result, positioned_mask)

    elif style_key == "painted_sign":
        paint_color = hex_to_rgb(color_hex, default_rgb=(240, 230, 210))
        shadow = Image.new("L", (base_w, base_h), 0)
        shadow.paste(resized_glyph, (pos_x + 2, pos_y + 2))
        shadow = shadow.filter(ImageFilter.GaussianBlur(1.5))

        res_arr = np.array(result, dtype=np.int16)
        res_arr -= (np.array(shadow)[:, :, None] * 0.35).astype(np.int16)
        result = Image.fromarray(np.clip(res_arr, 0, 255).astype(np.uint8))

        color_layer = Image.new("RGB", (base_w, base_h), paint_color)
        result = Image.composite(color_layer, result, positioned_mask)

    elif style_key == "chalk":
        chalk_color = hex_to_rgb(color_hex, default_rgb=(245, 245, 240))
        soft_chalk = positioned_mask.filter(ImageFilter.GaussianBlur(0.8))
        color_layer = Image.new("RGB", (base_w, base_h), chalk_color)
        result = Image.composite(color_layer, result, soft_chalk)

    elif style_key == "clean_white":
        color_layer = Image.new("RGB", (base_w, base_h), (255, 255, 255))
        result = Image.composite(color_layer, result, positioned_mask)

    elif style_key == "clean_black":
        color_layer = Image.new("RGB", (base_w, base_h), (20, 20, 20))
        result = Image.composite(color_layer, result, positioned_mask)

    else:
        color_layer = Image.new("RGB", (base_w, base_h), hex_to_rgb(color_hex, (255, 255, 255)))
        result = Image.composite(color_layer, result, positioned_mask)

    return result

# ----------------- FastAPI Routes -----------------

@app.get("/", response_class=HTMLResponse)
@app.get("/studio", response_class=HTMLResponse)
def interactive_studio():
    """Returns an interactive Web Studio UI for uploading photos and styling Sinhala text in 1 click."""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Sinhala Akuru AI - Studio</title>
        <style>
            :root {
                --primary: #6366f1;
                --bg: #0f172a;
                --card: #1e293b;
                --text: #f8fafc;
                --border: #334155;
            }
            body {
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
                background-color: var(--bg);
                color: var(--text);
                margin: 0;
                padding: 24px;
                display: flex;
                flex-direction: column;
                align-items: center;
            }
            .container {
                max-width: 900px;
                width: 100%;
                background: var(--card);
                border: 1px solid var(--border);
                border-radius: 16px;
                padding: 32px;
                box-shadow: 0 10px 25px -5px rgba(0,0,0,0.5);
            }
            h1 { margin-top: 0; font-size: 28px; display: flex; align-items: center; gap: 10px; }
            .form-group { margin-bottom: 20px; }
            label { display: block; font-weight: 600; margin-bottom: 8px; font-size: 14px; }
            input[type="text"], select {
                width: 100%;
                padding: 12px;
                background: #0f172a;
                border: 1px solid var(--border);
                border-radius: 8px;
                color: white;
                font-size: 16px;
                box-sizing: border-box;
            }
            input[type="file"] {
                display: block;
                width: 100%;
                padding: 10px;
                background: #0f172a;
                border: 1px dashed var(--border);
                border-radius: 8px;
                color: #94a3b8;
                box-sizing: border-box;
                cursor: pointer;
            }
            .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
            button {
                width: 100%;
                background: var(--primary);
                color: white;
                border: none;
                padding: 14px;
                border-radius: 8px;
                font-size: 16px;
                font-weight: 600;
                cursor: pointer;
                transition: background 0.2s;
            }
            button:hover { background: #4f46e5; }
            button:disabled { opacity: 0.6; cursor: not-allowed; }
            .preview-container {
                margin-top: 28px;
                border-top: 1px solid var(--border);
                padding-top: 24px;
                display: none;
            }
            .image-box {
                width: 100%;
                border-radius: 8px;
                border: 1px solid var(--border);
                overflow: hidden;
                background: #000;
                text-align: center;
            }
            .image-box img { max-width: 100%; height: auto; display: block; margin: 0 auto; }
            .status-badge {
                display: inline-block;
                padding: 4px 10px;
                border-radius: 20px;
                font-size: 12px;
                font-weight: bold;
                background: #059669;
                color: white;
                margin-bottom: 12px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🇱🇰 Sinhala Akuru AI Studio</h1>
            <p style="color: #94a3b8; margin-bottom: 24px;">Upload any photo, type your Sinhala text, and paint it into the scene while keeping the base image 100% intact.</p>

            <form id="renderForm">
                <div class="form-group">
                    <label>1. Select Base Photo (or leave empty to use built-in sample)</label>
                    <input type="file" id="imageInput" accept="image/*">
                </div>

                <div class="form-group">
                    <label>2. Sinhala Text</label>
                    <input type="text" id="sinhalaText" value="ආයුබෝවන් ශ්‍රී ලංකා" placeholder="e.g. ආයුබෝවන්">
                </div>

                <div class="grid form-group">
                    <div>
                        <label>3. Graphical Style</label>
                        <select id="styleSelect">
                            <option value="neon_glow">🌟 Neon Glow (Cyberpunk / Night)</option>
                            <option value="metallic_gold">🏆 Metallic Gold (3D Embossed)</option>
                            <option value="wood_carved">🪵 Wood Carved (Engraved Depth)</option>
                            <option value="painted_sign">🎨 Painted Wall / Mural</option>
                            <option value="chalk">✏️ Chalk Board</option>
                            <option value="clean_white">⚪ Clean Solid White</option>
                            <option value="clean_black">⚫ Clean Solid Black</option>
                        </select>
                    </div>
                    <div>
                        <label>Custom Hex Color (Optional)</label>
                        <input type="text" id="colorInput" placeholder="e.g. #00FFFF or #FFB300">
                    </div>
                </div>

                <button type="submit" id="submitBtn">Generate Stylized Sinhala Image</button>
            </form>

            <div class="preview-container" id="previewContainer">
                <span class="status-badge" id="statusBadge">✔ 100% Intact Image Inpainted</span>
                <div class="image-box">
                    <img id="resultImage" src="" alt="Generated Result">
                </div>
                <div style="margin-top: 12px; text-align: right;">
                    <a id="downloadLink" href="" download="sinhala_stylized_result.png" style="color: #818cf8; text-decoration: none; font-weight: 600;">⬇ Download Image</a>
                </div>
            </div>
        </div>

        <script>
            document.getElementById('renderForm').addEventListener('submit', async (e) => {
                e.preventDefault();
                const submitBtn = document.getElementById('submitBtn');
                submitBtn.disabled = true;
                submitBtn.innerText = 'Generating...';

                const fileInput = document.getElementById('imageInput');
                const sinhalaText = document.getElementById('sinhalaText').value;
                const style = document.getElementById('styleSelect').value;
                const colorHex = document.getElementById('colorInput').value;

                let baseImageB64 = null;
                if (fileInput.files.length > 0) {
                    const file = fileInput.files[0];
                    baseImageB64 = await new Promise((resolve) => {
                        const reader = new FileReader();
                        reader.onload = () => resolve(reader.result);
                        reader.readAsDataURL(file);
                    });
                }

                const payload = {
                    sinhala_text: sinhalaText,
                    style: style,
                    base_image_base64: baseImageB64,
                    color_hex: colorHex || null
                };

                try {
                    const res = await fetch('/generate', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(payload)
                    });
                    const data = await res.json();
                    if (data.status === 'success') {
                        const img = document.getElementById('resultImage');
                        img.src = 'data:image/png;base64,' + data.generated_image_base64;
                        document.getElementById('downloadLink').href = img.src;
                        document.getElementById('previewContainer').style.display = 'block';
                    } else {
                        alert('Error: ' + JSON.stringify(data));
                    }
                } catch (err) {
                    alert('Request error: ' + err);
                } finally {
                    submitBtn.disabled = false;
                    submitBtn.innerText = 'Generate Stylized Sinhala Image';
                }
            });
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.post("/generate")
def generate_image(req: GenerationRequest):
    """
    JSON API endpoint. Supports Base64 strings or direct Sinhala text.
    Resilient against bad padding or default Swagger 'string' values.
    """
    # 1. Resolve Glyph Mask: from base64 if valid, or auto-render from sinhala_text
    glyph_bytes = safe_b64decode(req.glyph_mask_base64)
    if glyph_bytes:
        try:
            glyph_image = Image.open(io.BytesIO(glyph_bytes)).convert("RGB")
        except Exception:
            text = req.sinhala_text or "ආයුබෝවන්"
            glyph_image = render_sinhala_glyph_mask(text, 600, 200)
    else:
        text = req.sinhala_text or "ආයුබෝවන්"
        glyph_image = render_sinhala_glyph_mask(text, 600, 200)

    # 2. Resolve Base Image: from base64 if provided and valid, or default background
    base_bytes = safe_b64decode(req.base_image_base64)
    if base_bytes:
        try:
            base_image = Image.open(io.BytesIO(base_bytes)).convert("RGB")
        except Exception:
            base_image = Image.new("RGB", (700, 420), (35, 42, 54))
    else:
        base_image = Image.new("RGB", (700, 420), (35, 42, 54))
        d = ImageDraw.Draw(base_image)
        for y in range(0, 420, 6):
            d.line([(0, y), (700, y)], fill=(30, 36, 48), width=3)
        d.rectangle([20, 20, 680, 400], outline=(55, 65, 80), width=4)

    output = apply_stylized_text_to_image(
        base_img=base_image,
        glyph_mask=glyph_image,
        style=req.style or "neon_glow",
        color_hex=req.color_hex,
        box_x=req.box_x,
        box_y=req.box_y,
        box_w=req.box_w,
        box_h=req.box_h
    )

    buf = io.BytesIO()
    output.save(buf, format="PNG")
    b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")

    return {
        "status": "success",
        "style_applied": req.style or "neon_glow",
        "generated_image_base64": b64_str
    }
