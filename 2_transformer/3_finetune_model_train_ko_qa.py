import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.optim import AdamW

# Step 1: 대화 데이터셋 정의
conversation_data = [
    {"question": "안녕", "answer": "안녕, 나는 LLM 이야. 너는 누구야?"},
    {"question": "너 뭐 할 수 있어?", "answer": "나는 다양한 질문에 답할 수 있어."},
    {"question": "날씨 어때?", "answer": "날씨는 내가 있는 곳에서는 확인할 수 없어."},
    {"question": "몇 살이야?", "answer": "나는 나이가 없지만, 최신 정보를 제공하려 노력해."},
    {"question": "너 좋아하는 건 뭐야?", "answer": "나는 데이터를 분석하고 정보를 제공하는 걸 좋아해."},
]

def gen_model(question_text): # 추론 함수
    messages = [{"role": "user", "content": question_text}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    encoded_input = tokenizer(
        prompt,
        return_tensors="pt",
    )

    input_ids = encoded_input["input_ids"].to(device)
    attention_mask = encoded_input["attention_mask"].to(device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=64, 
            num_beams=1,
            temperature=0.1, # 우리가 가르친 대로 똑같이 말하도록 창의성을 낮춥니다.
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    # 생성된 부분만 잘라서 디코딩 (입력 프롬프트 제외)
    generated_ids = output_ids[0][input_ids.shape[1]:]
    decoded_output = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return decoded_output
    

# Step 2: 데이터셋 클래스 정의
class ConversationDataset(Dataset):
    def __init__(self, tokenizer, data, max_length=128):
        self.tokenizer = tokenizer
        self.data = data
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # [핵심 수정 1] Qwen 공식 대화 템플릿 적용
        messages = [
            {"role": "user", "content": item["question"]},
            {"role": "assistant", "content": item["answer"]}
        ]
        
        # 학습용 전체 텍스트 (질문 + 답변)
        full_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        
        # Loss 계산에서 제외할 질문 부분의 길이를 알기 위해 질문만 따로 템플릿화
        prompt_messages = [{"role": "user", "content": item["question"]}]
        prompt_text = self.tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )
        
        tokens = self.tokenizer(
            full_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        labels = tokens["input_ids"].clone()
        
        # 1. 질문 부분의 토큰 길이를 구해서 그만큼 -100으로 덮어씌움
        prompt_tokens = self.tokenizer(prompt_text, return_tensors="pt")["input_ids"]
        p_len = prompt_tokens.shape[1]
        labels[0, :p_len] = -100
        
        # 2. 패딩 토큰들도 -100으로 덮어씌움
        labels[0, tokens["input_ids"][0] == self.tokenizer.pad_token_id] = -100
        
        return {
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0),
            "labels": labels.squeeze(0),  
        }

# Step 3: 토크나이저 및 데이터셋 준비
model_id = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)   
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

dataset = ConversationDataset(tokenizer, conversation_data)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True) 

# Step 4: 모델 준비 및 디바이스 할당
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForCausalLM.from_pretrained(
    model_id,  
    torch_dtype=torch.bfloat16,  
).to(device)

# Step 5: Optimizer 설정
optimizer = AdamW(model.parameters(), lr=5e-5) 

decoded_output = gen_model("안녕")
print(f"\n모델 학습 전 테스트\nInput: 안녕")
print(f"Output: {decoded_output}")

# Step 6: 학습 루프
epochs = 50 
model.train()
for epoch in range(epochs):
    for batch in dataloader:
        input_ids = batch["input_ids"].long().to(device)  
        attention_mask = batch["attention_mask"].long().to(device)
        labels = batch["labels"].long().to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)        
        
        optimizer.step()

    print(f"Epoch {epoch + 1}, Loss: {loss.item():.6f}")

# Step 7: 모델 저장
cur_module_folder = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(cur_module_folder, "conversation_qwen2.5_0.5b_model")
tokenizer_path = os.path.join(cur_module_folder, "conversation_qwen2.5_0.5b_tokenizer")
model.save_pretrained(model_path)
tokenizer.save_pretrained(tokenizer_path)

print(f"\nModel saved to {model_path}")

# Step 8: 추론 준비
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).to(device) # 학습할 때 썼던 bfloat16을 추론할 때도 똑같이 써줘야 에러가 안남.
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

# Step 9: 파인튜닝 모델 테스트
model.eval()

inference_text = "안녕"
decoded_output = gen_model(inference_text)
print(f"\n전체 파인튜닝(Full Fine-Tuning) 후 결과\nInput: {inference_text}")
print(f"Output: {decoded_output}")