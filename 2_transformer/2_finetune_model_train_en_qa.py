import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW

# Step 1: 대화 데이터셋 정의
conversation_data = [
    {"question": "Hello", "answer": "Hi, I'm an LLM student. Who are you?"},
    {"question": "What can you do?", "answer": "I can answer a variety of questions."},
    {"question": "How's the weather?", "answer": "The weather isn't available where I am."},
    {"question": "How old are you?", "answer": "I'm not old, but I try to provide up-to-date information."},
    {"question": "What do you like to do?", "answer": "I like analyzing data and providing information."}
]


# DistilBERT is a cost-effective BERT-family model for small VRAM.
MODEL_NAME = "distilbert-base-uncased"


label_to_answer = {i: item["answer"] for i, item in enumerate(conversation_data)}


def gen_model(question):
    # Intent classification -> fixed answer retrieval
    encoded_input = tokenizer(
        question,
        max_length=64,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    input_ids = encoded_input["input_ids"].to(device)
    attention_mask = encoded_input["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        predicted_label = torch.argmax(outputs.logits, dim=-1).item()

    output_text = label_to_answer[predicted_label]
    return output_text

# Step 2: 데이터셋 클래스 정의
class ConversationDataset(Dataset):
    def __init__(self, tokenizer, data, max_length=64):
        self.tokenizer = tokenizer
        self.data = data
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # Classify question into one of predefined answer labels.
        question_tokens = self.tokenizer(
            item["question"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": question_tokens["input_ids"].squeeze(0),
            "attention_mask": question_tokens["attention_mask"].squeeze(0),
            "labels": torch.tensor(idx, dtype=torch.long),  # 질문별 정답 클래스
        }

# Step 3: 토크나이저 및 데이터셋 준비
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
dataset = ConversationDataset(tokenizer, conversation_data)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Step 4: 모델 준비
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(conversation_data),
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Step 5: Optimizer 설정
optimizer = AdamW(model.parameters(), lr=5e-5)

decoded_output = gen_model("Hello")
print(f"\n모델 학습 전 테스트\nInput: Hello")
print(f"Output: {decoded_output}")

# Step 6: 학습 루프
epochs = 50
model.train()
for epoch in range(epochs):
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

# Step 7: 모델 저장
cur_module_folder = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(cur_module_folder, "conversation_bert_mlm_model")
tokenizer_path = os.path.join(cur_module_folder, "conversation_bert_tokenizer")
model.save_pretrained(model_path)
tokenizer.save_pretrained(tokenizer_path)

print(f"Model saved to {model_path}")
print(f"Tokenizer saved to {tokenizer_path}")

# Step 8: 파인튜닝 모델 및 토크나이저 로드
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
model.to(device)

# Step 9: 파인튜닝 모델 테스트
model.eval()

inference_text =  "Hello."
decoded_output = gen_model(inference_text)
print(f"\n전체 파인튜닝(Full Fine-Tuning) 후 결과\nInput: {inference_text}")
print(f"Output: {decoded_output}")