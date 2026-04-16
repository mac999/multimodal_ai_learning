import torch
from PIL import Image
import requests
import matplotlib.pyplot as plt
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

# 모델과 프로세서 로드 
model_id = "Qwen/Qwen2-VL-2B-Instruct"
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16, # bfloat16으로 로드하여 VRAM 절약 속도 향상
    device_map="auto"
)
processor = AutoProcessor.from_pretrained(model_id)

# 테스트 이미지 로드
img_url = "https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg"
image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')
query = "What is in the image? Describe it briefly." # 이미지에 대한 질문

# 표준 VLM 프롬프트 구성  
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": query}
        ]
    }
]

text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
) # 텍스트 프롬프트 렌더링

inputs = processor(
    text=[text],
    images=[image],
    padding=True,
    return_tensors="pt"
).to(model.device) # 텍스트와 이미지를 processor에 한 번에 넣어 텐서로 변환

# 모델 추론
gen_kwargs = {
    "max_new_tokens": 256,
    "temperature": 0.4,
    "do_sample": True,
    "top_p": 0.8
}

model.eval()
outputs = model.generate(**inputs, **gen_kwargs)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, outputs)
] 
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0] # 프롬프트 부분을 잘라내고 생성된 텍스트만 추출
print(f"\n[질문]: {query}\n[답변]: {generated_text}\n")

plt.figure(figsize=(8, 8))
plt.imshow(image)
plt.axis('off')
plt.title(generated_text, fontsize=10, wrap=True)
plt.show()

import os
cur_module_path = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(cur_module_path, "cat2.jpg")
image = Image.open(image_path).convert("RGB")

import matplotlib.pyplot as plt
plt.imshow(image)
plt.axis('off')
plt.show()

# 이미지에 대한 질문과 답변 생성
query = "What is in the image? Describe it briefly."
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": query}
        ]
    }
]
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
inputs = processor(
    text=[text],
    images=[image],
    padding=True,
    return_tensors="pt"
).to(model.device)
gen_kwargs = {
    "max_new_tokens": 256,
    "temperature": 0.4,
    "do_sample": True,
    "top_p": 0.8
}
model.eval()
outputs = model.generate(**inputs, **gen_kwargs)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, outputs)
]
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(f"\n[질문]: {query}\n[답변]: {generated_text}\n")