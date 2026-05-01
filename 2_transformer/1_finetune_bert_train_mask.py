import torch
from transformers import BertTokenizer, BertForMaskedLM
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from tqdm import tqdm

# 1. 데이터셋 정의
class MaskedDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=128, mask_probability=0.15):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mask_probability = mask_probability

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        # Tokenize input text
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        input_ids = encoding["input_ids"].squeeze(0)            # ex) text="Hello world" → input_ids=[101, 7592, 2088, 102, 0, 0, ...] (padded to max_length)
        padding_mask = encoding["attention_mask"].squeeze(0)    # ex) text="Hello world" → attention_mask=[1, 1, 1, 1, 0, 0, ...] (1 for real tokens, 0 for padding)

        # Create masked input and labels
        labels = input_ids.clone() # tokenize labels for loss calculation
        rand = torch.rand(input_ids.shape)
        mask_arr = (rand < self.mask_probability) * (input_ids != self.tokenizer.cls_token_id) * (input_ids != self.tokenizer.sep_token_id) * (input_ids != self.tokenizer.pad_token_id) # Create a boolean mask to determine which tokens can be masked, excluding special tokens ([CLS], [SEP], [PAD])

        input_ids[mask_arr] = self.tokenizer.mask_token_id  # Replace with [MASK]

        return {
            "input_ids": input_ids, # Masked tokenized input text
            "attention_mask": padding_mask,  # Attention mask for the input
            "labels": labels # Original tokenized input text (used as labels for loss calculation)
        }

# 2. 데이터 준비
texts = [
    "The quick brown fox jumps over the lazy dog.",
    "Artificial intelligence is transforming the world.",
    "BERT stands for Bidirectional Encoder Representations from Transformers."
]

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
dataset = MaskedDataset(texts, tokenizer)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# 3. 모델 및 옵티마이저 초기화
model = BertForMaskedLM.from_pretrained("bert-base-uncased")
optimizer = AdamW(model.parameters(), lr=5e-5)

# 4. 학습 루프
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

losses = []
epochs = 500
for epoch in range(epochs):
    model.train()

    for batch in dataloader:
        # Move data to device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        print(f"Epoch {epoch + 1} Loss: {loss.item():.4f}")
        losses.append(loss.item())

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

import matplotlib.pyplot as plt
plt.plot(losses)
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Training Loss Over Time")
plt.show()

# 학습된 모델 테스트
#
test_texts = [
    "The quick brown [MASK] jumps over the lazy dog.",
    "Artificial intelligence is [MASK] the world.",
    "BERT stands for [MASK] Encoder Representations from Transformers."
]

# 토크나이저로 테스트 데이터 인코딩
test_encodings = tokenizer(
    test_texts,
    max_length=128,
    padding="max_length",
    truncation=True,
    return_tensors="pt"
)

# 토큰 시퀀스와 어텐션 마스크를 디바이스로 이동
input_ids = test_encodings["input_ids"].to(device)
attention_mask = test_encodings["attention_mask"].to(device)

# 모델 평가
model.eval()

with torch.no_grad():
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    predictions = outputs.logits

# 평가 결과 출력
for i, test_text in enumerate(test_texts):
    mask_positions = input_ids[i] == tokenizer.mask_token_id    # [101, 200, 103, 300] → [False, False, True, False]
    mask_indices = mask_positions.nonzero(as_tuple=True)        # (tensor([2]),)
    mask_indices = mask_indices[0]      # tensor([2])
    masked_index = mask_indices.item()  # 2

    # mask 된 토큰의 예측된 단어 사전 중 가장 높은 확률을 가진 토큰 ID 추출
    logits = predictions[i, masked_index]               # shape: [vocab_size]. vocab_size=30522 (BERT)
    predicted_token_id_tensor = logits.argmax(dim=-1)   # ex) logits=[0.1, 0.2, 0.05, ..., 0.3] → predicted_token_id_tensor=7592
    predicted_token_id = predicted_token_id_tensor.item()   # 7592
    print(f"logits shape: {logits.shape}, logit: {logits[predicted_token_id]}, predicted_token_id: {predicted_token_id}")
    predicted_token = tokenizer.decode([predicted_token_id])                # 예측된 토큰 ID를 실제 단어로 디코딩

    print(f"Original: {test_text}")
    print(f"Prediction: {test_text.replace(tokenizer.mask_token, predicted_token)}") # [MASK] 토큰을 예측된 단어로 대체하여 출력
