import io
import os
import base64
import torch
from PIL import Image, ImageDraw, ImageFont
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler

app = FastAPI(
    title="Sinhala Akuru AI - Inference Server",
    description="FastAPI service for generating realistic images containing accurate Sinhala text using Glyph-Conditioned ControlNet",
    version="1.0.0"
)

pipeline = None
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

class GenerationRequest(BaseModel):
    scene_prompt: str = Field(..., description="Background scene description in English")
    glyph_mask_base64: str = Field(..., description="Base64-encoded PNG of white Sinhala glyphs on black background")
    controlnet_scale: float = Field(1.15, description="Conditioning scale for glyph preservation")
    steps: int = Field(25, description="Inference sampling steps")

@app.on_event("startup")
def load_pipeline():
    global pipeline
    checkpoint_path = "checkpoints/sinhala-glyph-controlnet"
    base_model = "runwayml/stable-diffusion-v1-5"

    print(f"Loading Sinhala Glyph Pipeline on {device}...")
    if os.path.exists(checkpoint_path):
        print(f"Loading custom trained ControlNet checkpoint from {checkpoint_path}...")
        controlnet = ControlNetModel.from_pretrained(checkpoint_path, torch_dtype=dtype)
    else:
        print(f"Warning: {checkpoint_path} not found. Initializing with default ControlNet for testing...")
        controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=dtype)

    try:
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
        print("[SUCCESS] Sinhala Glyph Diffusion pipeline loaded successfully.")
    except Exception as e:
        print(f"Pipeline loading error: {e}. Running in lightweight mock/fallback mode.")
        pipeline = None

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "device": device,
        "pipeline_loaded": pipeline is not None
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
        # Fallback preview mode when running without full weights/GPU
        output = Image.new("RGB", (512, 512), (40, 44, 52))
        draw = ImageDraw.Draw(output)
        draw.text((20, 20), "Sinhala Akuru AI (Preview Mode)", fill=(255, 215, 0))
        draw.text((20, 50), f"Prompt: {req.scene_prompt[:45]}...", fill=(200, 200, 200))
        # Blend glyph image over canvas
        blended = Image.blend(output, glyph_image, alpha=0.6)
        output = blended

    buf = io.BytesIO()
    output.save(buf, format="PNG")
    b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")

    return {
        "status": "success",
        "scene_prompt": req.scene_prompt,
        "generated_image_base64": b64_str
    }
