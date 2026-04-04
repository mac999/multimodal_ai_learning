import math
import torch, torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

# 1. LoRA 레이어 정의  
class LoRALinear(nn.Module):
    def __init__(self, original_layer, rank=32, alpha=64):
        super().__init__()
        self.original_layer = original_layer  # W0 (Frozen)
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank  # Scale Factor (논문의 alpha / r)

        in_features = original_layer.in_features
        out_features = original_layer.out_features

        # 로우-랭크 행렬 AB 정의 (W0 + Delta W)
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5)) # Kaiming uniform 초기화 사용 (실제 LoRA 코드에서 사용하는 방식. 안정적인 학습)

        # 원본 레이어의 가중치(weight) 뿐만 아니라 편향(bias)이 있다면 모두 프리즈
        if self.original_layer.bias is not None:
            self.original_layer.bias.requires_grad = False
        for param in self.original_layer.parameters():
            param.requires_grad = False

    def forward(self, x):
        base_output = self.original_layer(x) 

        # LoRA 연산: Delta W = (B @ A) * scaling
        lora_output = (x @ self.lora_A.t() @ self.lora_B.t()) * self.scaling
        
        return base_output + lora_output # 1) h = x * W0^T

# 2. 트랜스포머 블록
class TinyTransformerBlock(nn.Module):
    def __init__(self, d_model=64):
        super().__init__()
        
        # Attention 부분
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)

        # MLP (FFN) 부분. # FFN에서 d_model 4배로 차원 확장(논문 Attention is all you need 기준)
        self.up_proj = nn.Linear(d_model, d_model * 4)  
        self.down_proj = nn.Linear(d_model * 4, d_model)

    def forward(self, x):
        x = self.q_proj(x)
        x = self.v_proj(x) 
        
        x = self.up_proj(x)
        x = F.relu(x)  
        x = self.down_proj(x)
        
        return x

# 3. LoRA 인젝션 함수 정의  
def inject_lora_by_name(model, target_names, rank=32, alpha=64):
    targets = []
    for name, module in model.named_modules():
        matches = []
        for t in target_names:
            if t in name:
                matches.append(t)
        is_target = len(matches) > 0
        if is_target and isinstance(module, nn.Linear):
            targets.append((name, module))
                
    for name, module in targets: # 수집된 레이어 교체 진행
        parts = name.split('.') # 'layer1.attention.q_proj' > 부모 객체들 + 'q_proj'
        child_name = parts[-1]
        
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
            
        # setattr을 이용해 기존 nn.Linear를 아답터 레이어로 삽입
        setattr(parent, child_name, LoRALinear(module, rank=rank, alpha=alpha))
        print(f"'{name}'레이어를 LoRA(rank={rank})로 교체.")

# 4. 모델 학습
alphabet_in = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" # 학습 데이터 준비
alphabet_out = alphabet_in[::-1] # "ZYX...A"

d_model = 64
vocab_size = 26
embedding = nn.Embedding(vocab_size, d_model) # 임의 값으로 임베딩 레이어 생성

# 정적 텐서 생성 (Batch=1, Seq=26, Dim=64)
with torch.no_grad():
    x_indices = torch.tensor([ord(c) - ord('A') for c in alphabet_in]).unsqueeze(0)
    y_indices = torch.tensor([ord(c) - ord('A') for c in alphabet_out]).unsqueeze(0)
    
    input_x = embedding(x_indices) # 입력 벡터 (A-Z)
    target_y = embedding(y_indices) # 목표 벡터 (Z-A)

# 5. 모델 구축 및 LoRA 주입
d_model = 64
model = TinyTransformerBlock(d_model=d_model)
model.eval()
final_output = model(input_x)
def get_gimilarity(char_in, char_out):
    idx_in = ord(char_in.upper()) - ord('A')  # 문자를 인덱스로 변환 (A=0, B=1, ...)
    pred_vec = final_output[0, idx_in:idx_in+1] # 모델이 char_in 대해 예측한 벡터 추출 [1, 1, 64]

    idx_out = ord(char_out.upper()) - ord('A')
    with torch.no_grad(): # 비교 대상인 char_out의 실제 정규 임베딩 벡터 추출 [1, 64]
        target_vec = embedding.weight[idx_out:idx_out+1]

    return F.cosine_similarity(pred_vec, target_vec).item() # 두 벡터 간의 코사인 유사도 계산

def compare_similarity():
    print(f"입력과 목표 간의 유사도 (1.0에 가까울수록 유사)")
    print(f"입력 'A' -> 목표 'Z' 유사도: {get_gimilarity('A', 'Z'):.4f}")
    print(f"입력 'M' -> 목표 'N' 유사도: {get_gimilarity('M', 'N'):.4f}")
    print(f"입력 'Z' -> 목표 'A' 유사도: {get_gimilarity('Z', 'A'):.4f}")
    print(f"입력 'A' -> 목표 'A' 유사도: {get_gimilarity('A', 'A'):.4f}")
    print(f"입력 'A' -> 목표 'B' 유사도: {get_gimilarity('A', 'B'):.4f}")
compare_similarity()

target_layers = ['q_proj', 'v_proj', 'up_proj', 'down_proj']
inject_lora_by_name(model, target_layers, rank=16)

# 6. 학습 설정
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
print(f"모델 전체(Full Model) 학습 시작")
model.train()
EPOCHS = 200
for epoch in range(1, EPOCHS + 1):
    optimizer.zero_grad()
    output = model(input_x) 
    loss = F.mse_loss(output, target_y)
    loss.backward()
    
    # Gradient clipping으로 gradient explosion 방지
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f"Epoch [{epoch:4d}/{EPOCHS}] | Loss: {loss.item():.6f}")

# 7. 최종 검증
model.eval()
final_output = model(input_x)
def get_gimilarity(char_in, char_out):
    idx_in = ord(char_in.upper()) - ord('A')  # 문자를 인덱스로 변환 (A=0, B=1, ...)
    pred_vec = final_output[0, idx_in:idx_in+1] # 모델이 char_in 대해 예측한 벡터 추출 [1, 1, 64]

    idx_out = ord(char_out.upper()) - ord('A')
    with torch.no_grad(): # 비교 대상인 char_out의 실제 정규 임베딩 벡터 추출 [1, 64]
        target_vec = embedding.weight[idx_out:idx_out+1]

    return F.cosine_similarity(pred_vec, target_vec).item() # 두 벡터 간의 코사인 유사도 계산

compare_similarity()
