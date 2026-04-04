import torch
import torch.nn as nn
import torch.optim as optim

# 1. 학습 데이터 
data = [
    ("두 숫자를 더하는 함수", "def add(a, b):\n    return a + b"),
    ("헬로 월드를 출력해", 'print("Hello, World!")'),
    ("리스트를 정렬해", "list.sort()"),
    ("안녕이라고 말해", 'print("Hi")'),
]

# 2. 문자 단위 토큰화 (Character-level Tokenizer)
# 모든 문자를 모아 사전(Vocabulary) 생성
all_chars = sorted(list(set("".join([src + tgt for src, tgt in data]) + "<s></s><p>")))
char_to_int = {c: i for i, c in enumerate(all_chars)}
int_to_char = {i: c for i, c in enumerate(all_chars)}
vocab_size = len(all_chars)

def encode(s): return [char_to_int[c] for c in s]
def decode(l): return "".join([int_to_char[i] for i in l])

class TinyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=2):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = nn.Parameter(torch.zeros(1, 500, d_model))  # 위치 인코딩
        # PyTorch 표준 Transformer (인코더-디코더 구조)
        self.transformer = nn.Transformer(
            d_model=d_model, nhead=nhead, 
            num_encoder_layers=num_layers, 
            num_decoder_layers=num_layers,
            batch_first=True,
            dim_feedforward=512  # FFN 크기 증가
        )
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt):
        # 임베딩 + 위치 인코딩
        src_emb = self.embed(src) + self.pos_encoder[:, :src.size(1), :]
        tgt_emb = self.embed(tgt) + self.pos_encoder[:, :tgt.size(1), :]
        
        # Causal mask 생성 (미래 토큰을 보지 못하도록)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1)).to(src.device)
        
        out = self.transformer(src_emb, tgt_emb, tgt_mask=tgt_mask)
        return self.fc_out(out)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TinyTransformer(vocab_size).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0003)  # 학습률 낮춤
criterion = nn.CrossEntropyLoss()

# 간단한 학습 루프
for epoch in range(300):  # 에폭 증가
    total_loss = 0
    for src_text, tgt_text in data:
        model.train()
        # 입력과 출력 준비
        src = torch.tensor([encode(src_text)]).to(device)
        tgt_full = encode("<s>" + tgt_text + "</s>")
        tgt_input = torch.tensor([tgt_full[:-1]]).to(device) # 입력: <s> ...
        tgt_output = torch.tensor([tgt_full[1:]]).to(device) # 정답: ... </s>

        output = model(src, tgt_input)
        loss = criterion(output.view(-1, vocab_size), tgt_output.view(-1))
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 그래디언트 클리핑
        optimizer.step()
        total_loss += loss.item()
    if (epoch+1) % 10 == 0: print(f"Epoch {epoch+1}, Loss: {total_loss/len(data):.4f}")

# 한 글자씩 생성하는 테스트
def generate_code(prompt, max_length=50):
    model.eval()
    src = torch.tensor([encode(prompt)]).to(device)
    generated = encode("<s>")
    
    for _ in range(max_length):
        tgt_input = torch.tensor([generated]).to(device)
        with torch.no_grad():
            output = model(src, tgt_input)
            next_char_idx = output[0, -1, :].argmax().item()  # 가장 확률 높은 문자 선택
            generated.append(next_char_idx)
            if int_to_char[next_char_idx] == ">":  # '>' 체크
                if len(generated) >= 3 and int_to_char[generated[-2]] == 's' and int_to_char[generated[-3]] == '/':
                    break
            
    return decode(generated).replace("<s>", "").replace("</s>", "")

print("\n--- 생성 결과 ---")
print(f"입력: 헬로 월드를 출력해 -> 결과: {generate_code('헬로 월드를 출력해')}")