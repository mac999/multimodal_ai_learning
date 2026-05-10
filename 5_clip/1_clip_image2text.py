# you should install the CLIP library first
# pip install git+https://github.com/openai/CLIP.git
import os
import clip
import torch
from torchvision.datasets import CIFAR100

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)

# Download the dataset
cifar100 = CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=False)

# Prepare the inputs
image, class_id = cifar100[3637]
print(f"Class ID: {class_id}, Class Name: {cifar100.classes[class_id]}")

import matplotlib.pyplot as plt

# Display the image
plt.imshow(image)
plt.axis('off')
plt.title(f"Class: {cifar100.classes[class_id]}")
plt.show()

image_input = preprocess(image).unsqueeze(0).to(device)
text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in cifar100.classes]).to(device)

# Calculate features
with torch.no_grad():
    image_features = model.encode_image(image_input)
    text_features = model.encode_text(text_inputs)

# Pick the top 5 most similar labels for the image
image_features /= image_features.norm(dim=-1, keepdim=True)
text_features /= text_features.norm(dim=-1, keepdim=True)
similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
values, indices = similarity[0].topk(5)

# Print the result
print("\nTop predictions:\n")
for value, index in zip(values, indices):
    print(f"{cifar100.classes[index]:>16s}: {100 * value.item():.2f}%")

# 테스트 이미지(Zeroshot)로 CLIP 모델 예측 예제
import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

fname = os.path.join(os.path.dirname(__file__), 'cat3.jpg')  # Zeroshot 예제 이미지 경로 (Unseen 이미지)
file = Image.open(fname)
image = preprocess(file).unsqueeze(0).to(device)

# 1) 기존 3개 분류
print("1. 기존 3개 클래스 분류")
labels = ["big dog", "siamese cat", "orange tabby cat"]
text = clip.tokenize(labels).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    
    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

for i in range(len(labels)):
    print(f"  {labels[i]:<20s}: {probs[0][i]*100:.2f}%")

# 2) 키워드 중심 간략 설명
print("2. 키워드 기반 이미지 설명")

# 임의의 키워드 조합으로 descriptions 생성
descriptions = ["cat", "dog", "kitten", "orange cat", "white cat", "gray cat", "brown cat", 
                "cat sitting", "cat lying", "dog sitting", "cat indoors", "cat outdoors"]

text = clip.tokenize(descriptions).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    
    # 유사도 계산
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    
    # 상위 5개 결과
    values, indices = similarity[0].topk(5)

descriptions_text = [descriptions[i] for i in indices]
print(descriptions_text)

# 이미지 표시
plt.figure(figsize=(8, 6))
plt.imshow(file)
plt.axis('off')
plt.title(f"{descriptions_text}", fontsize=12, pad=10)
plt.tight_layout()
plt.show()

