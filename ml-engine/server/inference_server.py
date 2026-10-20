import io
import os
import base64
from PIL import Image, ImageDraw, ImageFont
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(
    title="Sinhala Akuru AI - Inference Server",
    description="FastAPI service for generating realistic images containing accurate Sinhala text using Glyph-Conditioned ControlNet",
    version="1.0.0"
)

pipeline = None
device = "cpu"
dtype = None

class GenerationRequest(BaseModel):
    scene_prompt: str = Field(..., description="Background scene description in English")
    glyph_mask_base64: str = Field(..., description="Base64-encoded PNG of white Sinhala glyphs on black background")
    controlnet_scale: float = Field(1.15, description="Conditioning scale for glyph preservation")
    steps: int = Field(25, description="Inference sampling steps")

@app.on_event("startup")
def load_pipeline():
    global pipeline, device, dtype
    print("[INFO] Initializing Sinhala Akuru AI Inference Server...")

    try:
        import torch
        from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        checkpoint_path = "checkpoints/sinhala-glyph-controlnet"
        base_model = "runwayml/stable-diffusion-v1-5"

        print(f"[INFO] PyTorch detected (device: {device}). Checking checkpoints...")
        if os.path.exists(checkpoint_path):
            print(f"[INFO] Loading custom ControlNet from {checkpoint_path}...")
            controlnet = ControlNetModel.from_pretrained(checkpoint_path, torch_dtype=dtype)
        else:
            print("[INFO] No custom trained checkpoint found yet. Loading base ControlNet...")
            controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=dtype)

        pipeline = StableDiffusionControlNetPipeline.from_pretrained(
            base_model,
            controlnet=controlnet,
            torch_dtype=dtype
        )
        pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)

        if device == "cuda":
            pipeline.enable_model_cpu_offload()
        else:
            pipeline.to("cpu")
        print("[SUCCESS] Full Sinhala Glyph Diffusion pipeline loaded successfully.")

    except ImportError as e:
        print(f"[NOTE] Deep learning libraries not installed ({e}).")
        print("       Server will run in Preview / Testing Mode.")
        pipeline = None
    except Exception as e:
        print(f"[NOTE] Pipeline initialization info: {e}")
        print("       Server is running in Preview Mode.")
        pipeline = None

@app.get("/")
def root():
    return {
        "message": "Sinhala Akuru AI Inference Server is running",
        "docs_url": "/docs",
        "health_url": "/health"
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "device": device,
        "mode": "full_diffusion" if pipeline is not None else "preview_mode"
    }

@app.post("/generate")
def generate_image(req: GenerationRequest):
    try:
        glyph_bytes = base64.b64decode(req.glyph_mask_base64)
        glyph_image = Image.open(io.BytesIO(glyph_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 glyph mask: {str(e)}")

    if pipeline is not None:
        try:
            output = pipeline(
                prompt=req.scene_prompt,
                image=glyph_image,
                num_inference_steps=req.steps,
                controlnet_conditioning_scale=req.controlnet_scale
            ).images[0]
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Diffusion inference failure: {str(e)}")
    else:
        # Realistic preview composite mode when testing without full GPU weights
        img_size = glyph_image.size
        # Create atmospheric background
        output = Image.new("RGB", img_size, (28, 32, 40))
        draw = ImageDraw.Draw(output)

        # Draw decorative banner header
        draw.rectangle([0, 0, img_size[0], 50], fill=(20, 24, 30))
        draw.text((15, 15), "[Sinhala Akuru AI - Preview Composite]", fill=(255, 215, 0))

        # Composite the glyph mask with warm golden glow
        tinted_glyph = Image.new("RGB", img_size, (0, 0, 0))
        # Where glyph is white, tint it gold/cyan
        glyph_gray = glyph_image.convert("L")
        gold_color = Image.new("RGB", img_size, (255, 220, 100))
        tinted_glyph.paste(gold_color, mask=glyph_gray)

        # Blend
        output = Image.blend(output, tinted_glyph, alpha=0.85)
        draw = ImageDraw.Draw(output)
        draw.text((15, img_size[1] - 35), f"Scene: {req.scene_prompt[:50]}", fill=(180, 190, 205))

    buf = io.BytesIO()
    output.save(buf, format="PNG")
    b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")

    return {
        "status": "success",
        "scene_prompt": req.scene_prompt,
        "mode": "diffusion" if pipeline is not None else "preview",
        "generated_image_base64": b64_str
    }
