import os
import json
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DDPMScheduler,
    UNet2DConditionModel,
)
from transformers import CLIPTextModel, CLIPTokenizer

class SinhalaGlyphDataset(Dataset):
    """
    Dataset loader for Sinhala glyph conditioning.
    Pairs (Scene Prompt, Target Photo, Glyph Condition Image, Latent Region Mask).
    """
    def __init__(self, data_dir: str, tokenizer: CLIPTokenizer):
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        meta_path = os.path.join(data_dir, "metadata.jsonl")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Missing {meta_path}. Run generate_synthetic_data.py first.")

        with open(meta_path, "r", encoding="utf-8") as f:
            self.items = [json.loads(line) for line in f]

        self.img_tf = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        self.cond_tf = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor()
        ])
        self.mask_tf = transforms.Compose([
            transforms.Resize((64, 64)),  # Match latent space resolution (512 / 8)
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        item = self.items[idx]
        target = Image.open(os.path.join(self.data_dir, item["target_path"])).convert("RGB")
        glyph = Image.open(os.path.join(self.data_dir, item["glyph_path"])).convert("RGB")
        mask = Image.open(os.path.join(self.data_dir, item["mask_path"])).convert("L")

        tokens = self.tokenizer(
            item["prompt"],
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt"
        )
        return {
            "pixel_values": self.img_tf(target),
            "conditioning_pixel_values": self.cond_tf(glyph),
            "region_mask": self.mask_tf(mask),
            "input_ids": tokens.input_ids.squeeze(0)
        }

def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on device: {device}")
    if device == "cpu":
        print("Note: Running on CPU. For practical training, an NVIDIA GPU with >= 6GB VRAM is recommended.")

    dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"Loading base model components: {args.base_model}...")
    tokenizer = CLIPTokenizer.from_pretrained(args.base_model, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.base_model, subfolder="text_encoder").to(device, dtype=dtype)
    vae = AutoencoderKL.from_pretrained(args.base_model, subfolder="vae").to(device, dtype=dtype)
    unet = UNet2DConditionModel.from_pretrained(args.base_model, subfolder="unet").to(device, dtype=dtype)

    # Initialize ControlNet from UNet weights to learn Sinhala glyph conditioning
    print("Initializing ControlNet from UNet...")
    controlnet = ControlNetModel.from_unet(unet).to(device)
    if device == "cuda":
        controlnet.enable_gradient_checkpointing()

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    controlnet.train()

    noise_scheduler = DDPMScheduler.from_pretrained(args.base_model, subfolder="scheduler")
    optimizer = torch.optim.AdamW(controlnet.parameters(), lr=args.learning_rate, weight_decay=1e-2)

    dataset = SinhalaGlyphDataset(args.data_dir, tokenizer)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    print(f"Starting training: {args.epochs} epochs, {len(dataset)} samples, lambda={args.region_weight_lambda}")

    for epoch in range(args.epochs):
        for step, batch in enumerate(loader):
            optimizer.zero_grad()

            with torch.no_grad():
                latents = vae.encode(batch["pixel_values"].to(device, dtype=dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
                encoder_hidden_states = text_encoder(batch["input_ids"].to(device))[0]

            noise = torch.randn_like(latents, dtype=torch.float32)
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=device
            ).long()
            noisy_latents = noise_scheduler.add_noise(latents.float(), noise, timesteps)

            cond_img = batch["conditioning_pixel_values"].to(device, dtype=torch.float32)
            down_samples, mid_sample = controlnet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=encoder_hidden_states.float(),
                controlnet_cond=cond_img,
                return_dict=False
            )

            if device == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    model_pred = unet(
                        noisy_latents.half(),
                        timesteps,
                        encoder_hidden_states=encoder_hidden_states,
                        down_block_additional_residuals=[s.half() for s in down_samples],
                        mid_block_additional_residual=mid_sample.half()
                    ).sample
            else:
                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    down_block_additional_residuals=down_samples,
                    mid_block_additional_residual=mid_sample
                ).sample

            # Region-Weighted Loss: amplify loss by (1 + lambda * M) inside text box
            mask = batch["region_mask"].to(device, dtype=torch.float32)
            pixel_weights = 1.0 + (args.region_weight_lambda * mask)
            loss = (pixel_weights * (model_pred.float() - noise) ** 2).mean()

            loss.backward()
            optimizer.step()

            if step % 25 == 0:
                print(f"Epoch {epoch + 1}/{args.epochs} | Step {step}/{len(loader)} | Region-Weighted Loss: {loss.item():.4f}")

    os.makedirs(args.output_dir, exist_ok=True)
    controlnet.save_pretrained(args.output_dir)
    print(f"✔ Saved trained Sinhala Glyph ControlNet checkpoint to: {args.output_dir}")

def parse_args():
    parser = argparse.ArgumentParser(description="Train ControlNet for Sinhala visual text rendering")
    parser.add_argument("--base_model", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--data_dir", type=str, default="dataset")
    parser.add_argument("--output_dir", type=str, default="checkpoints/sinhala-glyph-controlnet")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--region_weight_lambda", type=float, default=5.0, help="Loss multiplier inside text region")
    return parser.parse_args()

if __name__ == "__main__":
    train(parse_args())
