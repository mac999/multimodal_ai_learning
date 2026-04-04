"""
Stable Diffusion - Hugging Face 모델 사용
Model: runwayml/stable-diffusion-v1-5 (4GB VRAM)
"""

import torch
from diffusers import StableDiffusionPipeline
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

print(f"Device: {DEVICE}")

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    safety_checker=None,
)
pipe = pipe.to(DEVICE)

if DEVICE == "cuda":
    pipe.enable_attention_slicing()

print("Model loaded!\n")

def generate_image(prompt, negative_prompt="", num_images=1, steps=50, guidance_scale=7.5, seed=None):
    generator = None
    if seed is not None:
        generator = torch.Generator(device=DEVICE).manual_seed(seed)
    
    images = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_images_per_prompt=num_images,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        generator=generator,
        height=512,
        width=512,
    ).images
    
    return images

# Example 1: Unicorn
prompt = "a magical unicorn in an enchanted forest, fantasy art, detailed, beautiful lighting, 4k"
negative_prompt = "blurry, low quality, distorted"

images = generate_image(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_images=4,
    steps=30,
    guidance_scale=7.5,
    seed=42
)

fig, axes = plt.subplots(2, 2, figsize=(10, 10))
for i, img in enumerate(images):
    row, col = i // 2, i % 2
    axes[row, col].imshow(img)
    axes[row, col].set_title(f"Unicorn {i+1}", fontsize=10)
    axes[row, col].axis('off')

plt.suptitle(f'Prompt: "{prompt}"', fontsize=11, wrap=True)
plt.tight_layout()
save_path = os.path.join(SCRIPT_DIR, 'sd_unicorn.png')
plt.savefig(save_path, dpi=200, bbox_inches='tight')
plt.show()

# Example 2: Various prompts
prompts = [
    "a cute cat wearing a wizard hat, digital art",
    "a futuristic city at sunset, cyberpunk style",
    "a dragon flying over mountains, fantasy painting",
    "a peaceful beach with palm trees, tropical paradise"
]

fig, axes = plt.subplots(2, 2, figsize=(12, 12))
for i, prompt in enumerate(prompts):
    img = generate_image(
        prompt=prompt,
        negative_prompt="blurry, low quality",
        num_images=1,
        steps=25,
        guidance_scale=7.5,
        seed=i*10
    )[0]
    
    row, col = i // 2, i % 2
    axes[row, col].imshow(img)
    axes[row, col].set_title(prompt[:50] + "...", fontsize=9)
    axes[row, col].axis('off')

plt.suptitle('Stable Diffusion Various Examples', fontsize=12)
plt.tight_layout()
save_path = os.path.join(SCRIPT_DIR, 'sd_examples.png')
plt.savefig(save_path, dpi=200, bbox_inches='tight')
plt.show()

# Example 3: CFG comparison
prompt = "a majestic lion with golden mane, photorealistic"
cfg_scales = [1.0, 3.0, 7.5, 15.0]

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for i, cfg in enumerate(cfg_scales):
    img = generate_image(
        prompt=prompt,
        num_images=1,
        steps=25,
        guidance_scale=cfg,
        seed=123
    )[0]
    
    axes[i].imshow(img)
    axes[i].set_title(f'CFG={cfg}', fontsize=10)
    axes[i].axis('off')

plt.suptitle(f'CFG Scale Effect: "{prompt}"', fontsize=11)
plt.tight_layout()
save_path = os.path.join(SCRIPT_DIR, 'sd_cfg_comparison.png')
plt.savefig(save_path, dpi=200, bbox_inches='tight')
plt.show()

# Example 4: High quality single image
prompt = """
a mystical unicorn with a flowing rainbow mane standing in an enchanted forest,
magical sparkles, moonlight, fantasy art, highly detailed, beautiful composition,
trending on artstation, 8k resolution
"""

negative_prompt = """
blurry, low quality, bad anatomy, distorted, ugly, deformed, 
low resolution, text, watermark
"""

img = generate_image(
    prompt=prompt.strip(),
    negative_prompt=negative_prompt.strip(),
    num_images=1,
    steps=50,
    guidance_scale=8.0,
    seed=999
)[0]

plt.figure(figsize=(8, 8))
plt.imshow(img)
plt.title("High Quality Unicorn (50 steps, CFG=8.0)", fontsize=12)
plt.axis('off')
plt.tight_layout()
save_path = os.path.join(SCRIPT_DIR, 'sd_hq_unicorn.png')
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.show()

print("Done!")

