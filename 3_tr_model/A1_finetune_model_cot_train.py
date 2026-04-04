import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM # BERT -> Auto로 변경
from torch.optim import AdamW

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Step 1: Load Pre-trained Tokenizer and Add Custom Token
model_id = "Qwen/Qwen2.5-0.5B" # Qwen 모델로 변경
tokenizer = AutoTokenizer.from_pretrained(model_id)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

custom_token = "[CoT]"
tokenizer.add_tokens([custom_token])  
print(f"Added token: {custom_token} -> ID: {tokenizer.convert_tokens_to_ids(custom_token)}")

# Step 2: Load Pre-trained Model and Resize Token Embeddings
# CausalLM 로드 및 bfloat16 적용 (NaN 방지)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)
model.resize_token_embeddings(len(tokenizer))
model.to(device)

# Step 3: Define CoT Dataset (Causal LM 방식에 맞게 전면 수정)
class CoTDataset(Dataset):
    def __init__(self, tokenizer, texts, max_length=128):
        self.tokenizer = tokenizer
        self.texts = texts
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        # 1. 문장 끝에 확실하게 종료 토큰 추가
        text = self.texts[idx] + self.tokenizer.eos_token
        
        encodings = self.tokenizer(
            text, 
            max_length=self.max_length, 
            padding="max_length", 
            truncation=True, 
            return_tensors="pt"
        )
        
        input_ids = encodings["input_ids"].squeeze(0)
        attention_mask = encodings["attention_mask"].squeeze(0)
        labels = input_ids.clone()

        # 2. [CoT] 토큰의 위치를 찾아 그 이전(질문 부분)은 학습(Loss)에서 제외
        cot_token_id = self.tokenizer.convert_tokens_to_ids("[CoT]")
        cot_indices = (input_ids == cot_token_id).nonzero(as_tuple=True)[0]
        
        if len(cot_indices) > 0:
            labels[:cot_indices[0] + 1] = -100 
            
        # [핵심 수정] pad_token_id 대신 attention_mask가 0인 부분(진짜 패딩)만 -100 처리!
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,  
        }

# Step 4: Sample CoT Dataset
texts = [
    "Question: What is 2 + 2? [CoT] Reasoning: First, we know that adding 2 to 2 gives 4. Final Answer: 4.",
    "Question: What is 5 * 3? [CoT] Reasoning: Multiply 5 by 3 to get 15. Final Answer: 15.",
    "Question: What is 10 - 3? [CoT] Reasoning: Subtract 3 from 10 to get 7. Final Answer: 7.",
    "Question: What is 7 + 5? [CoT] Reasoning: Adding 7 and 5 gives 12. Final Answer: 12.",
]

dataset = CoTDataset(tokenizer, texts)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Step 5: Define Optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

# Step 6: Training Loop
model.train()
epochs = 50 # 50번이면 충분히 외웁니다.
for epoch in range(epochs):
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # Forward Pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        # Backward Pass and Optimization
        optimizer.zero_grad()
        loss.backward()
        
        # 기울기 폭발(NaN) 방지
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) 
        
        optimizer.step()

    print(f"Epoch: {epoch + 1}. Loss: {loss.item():.6f}")

# Step 7: Save Fine-Tuned Model and Tokenizer
cur_module_folder = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(cur_module_folder, "cot_qwen_model")
tokenizer_path = os.path.join(cur_module_folder, "cot_qwen_tokenizer")

model.save_pretrained(model_path)
tokenizer.save_pretrained(tokenizer_path)

print(f"\nModel saved to {model_path}")
print(f"Tokenizer saved to {tokenizer_path}")

# Step 8: Load Fine-Tuned Model and Tokenizer for Inference
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
model.to(device)

# Step 9: Inference with Fine-Tuned Model
model.eval()

# Example inference text
inference_text = "Question: What is 10 - 3? [CoT]"

# Tokenize input
encoded_input = tokenizer(inference_text, return_tensors="pt")
input_ids = encoded_input["input_ids"].to(device)
attention_mask = encoded_input["attention_mask"].to(device)

# Causal LM은 generate() 함수를 사용해 뒤를 이어서 씁니다.
with torch.no_grad():
    output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=50, 
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id, # [안전장치] 확실히 여기서 멈추도록 명시
            do_sample=False, 
        )

# 입력 프롬프트를 제외하고 새롭게 생성된 부분만 잘라내기
generated_ids = output_ids[0][input_ids.shape[1]:]
decoded_output = tokenizer.decode(generated_ids, skip_special_tokens=True)

print(f"\nInput: {inference_text}")
print(f"Output:{decoded_output}")