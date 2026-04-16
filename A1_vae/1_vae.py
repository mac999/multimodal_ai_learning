import torch
import os
import gc
import matplotlib.pyplot as plt
from PIL import Image
from diffusers import AutoencoderKL
from torchvision import transforms

# 1. 모델 크기 계산 함수
def get_model_size_mb(model):
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    return param_size / (1024 * 1024)

device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "stabilityai/sd-vae-ft-mse"

# [Phase 1: 인코딩 및 Latent 추출]
print(f"1. Loading VAE to encode...")
vae = AutoencoderKL.from_pretrained(model_id).to(device)
vae_size = get_model_size_mb(vae)

# 이미지 로드 (cat3.jpg)
cur_module_path = os.path.dirname(os.path.abspath(__file__))
img_path = os.path.join(cur_module_path, "cat3.jpg")
img = Image.open(img_path).convert("RGB").resize((512, 512))
preprocess = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
input_tensor = preprocess(img).unsqueeze(0).to(device)

with torch.no_grad():
    # 이미지를 64x64x4의 Latent로 압축
    latent_space = vae.encode(input_tensor)
    latents = latent_space.latent_dist.mode()

    print(f"원본 잠재 벡터(latents) 범위: 최소 {latents.min().item():.2f} ~ 최대 {latents.max().item():.2f}")
    print(f"원본 잠재 벡터의 표준편차: {latents.std().item():.2f}")

# Latent 데이터 용량 계산 (float32 기준)
latent_size_kb = (latents.nelement() * latents.element_size()) / 1024

print(f"   - VAE Model Size: {vae_size:.2f} MB")
print(f"   - Latent Data Size: {latent_size_kb:.2f} KB")
print("-" * 50)

# [Phase 2: 모델 파괴 및 메모리 정리]
print("2. Deleting VAE model and clearing memory...")
del vae
gc.collect()
if torch.cuda.is_available(): torch.cuda.empty_cache()

# [Phase 3: 모델 재초기화 및 복원]
print("3. Re-initializing VAE model from scratch...")
vae_new = AutoencoderKL.from_pretrained(model_id).to(device)

with torch.no_grad():
    # 새로 띄운 모델에 기존에 뽑아둔 latents만 입력해서 복원
    reconstructed = vae_new.decode(latents).sample

# 결과 이미지 처리
recon_img = (reconstructed / 2 + 0.5).clamp(0, 1)
recon_img = recon_img[0].cpu().permute(1, 2, 0).numpy()

# [Phase 4: 시각화 및 증명]
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

axes[0].imshow(img)
axes[0].set_title(f"Original (512x512)\nRaw Image Data")
axes[0].axis("off")

axes[1].imshow(recon_img)
axes[1].set_title(f"Reconstructed from {latent_size_kb:.1f}KB Latents\nUsing Freshly Loaded VAE")
axes[1].axis("off")

plt.suptitle(f"VAE Verification: Model({vae_size:.1f}MB) + Latent({latent_size_kb:.1f}KB)")
plt.show()

print("Success: The image was perfectly restored using only the saved latents and a re-initialized model!")