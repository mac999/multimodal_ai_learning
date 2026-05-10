# VLM 교사 지도 감독 파인튜닝
import torch
from PIL import Image
import requests
import matplotlib.pyplot as plt
from transformers import AutoProcessor, BlipForConditionalGeneration

device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "Salesforce/blip-image-captioning-base" # https://huggingface.co/Salesforce/blip-image-captioning-base
learning_rate = 5e-5
epochs = 100

# VLM 모델과 프로세서 로드
print("모델 로드 중...")
processor = AutoProcessor.from_pretrained(model_id)
model = BlipForConditionalGeneration.from_pretrained(model_id).to(device)

# 실습용 온라인 이미지 로드 (강아지가 해변에 있는 사진)
img_url = "https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg"
image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')

plt.imshow(image)
plt.axis('off')
plt.title("Input Image")
plt.show()

# 학습 전 모델 추론 (Baseline)
model.eval()
inputs = processor(image, return_tensors="pt").to(device)

with torch.no_grad():
    out_ids = model.generate(**inputs, repetition_penalty=1.5, no_repeat_ngram_size=3)
    before_text = processor.decode(out_ids[0], skip_special_tokens=True)

print("[학습 전 모델 추론]")
print(f"원본 예측: {before_text}")

# 커스텀 데이터 준비 (파인튜닝 목표 데이터 설정)
target_text = "A magical photo of a golden dog relaxing on the beautiful beach."
print(f"학습 목표(Target): {target_text}\n")

# Vision Encoder Freeze
# 이미지 인코더는 이미 충분히 학습되어 있으므로 고정하고, 캡션을 생성하는 텍스트 디코더 파라미터만 업데이트함.
# 학습 파라미터 수 대폭 감소, 단일 샘플 과적합 억제함.
for param in model.vision_model.parameters():
    param.requires_grad = False
model.train() # 텍스트 디코더 파라미터만 업데이트하도록 설정

# 고정되지 않은 파라미터(텍스트 디코더)만 옵티마이저에 전달
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

# Cosine LR Scheduler 설정. 학습률을 코사인 곡선으로 점진 감소. 후반부 overshooting 방지
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

# 학습용 입력 데이터 전처리
train_inputs = processor(images=image, text=target_text, return_tensors="pt", padding=True).to(device)
train_inputs["labels"] = train_inputs["input_ids"].clone()

# VLM 모델 학습(SFT with Teacher Forcing)
print("[모델 파인튜닝 시작]")
for epoch in range(epochs):
    outputs = model(**train_inputs)
    loss = outputs.loss

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    scheduler.step()

    print(f"Epoch {epoch + 1}/{epochs} - Loss: {loss.item():.4f}  LR: {scheduler.get_last_lr()[0]:.2e}")

print("[모델 파인튜닝 완료]")

# 학습 후 모델 추론 (결과 확인)
model.eval()
with torch.no_grad():
    test_inputs = processor(image, return_tensors="pt").to(device)
    out_ids = model.generate(**test_inputs, repetition_penalty=1.5, no_repeat_ngram_size=3)
    after_text = processor.decode(out_ids[0], skip_special_tokens=True)

print("[학습 후 모델 추론]")
print(f"학습된 예측: {after_text}")
