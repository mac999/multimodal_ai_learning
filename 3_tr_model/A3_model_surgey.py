import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict

# 1. 어댑터 클래스
class ResidualAdapter(nn.Module):
    def __init__(self, original_layer, bottleneck_dim=4):
        super().__init__()
        self.original_layer = original_layer
        
        in_dim = original_layer.in_features
        out_dim = original_layer.out_features
        
        self.adapter = nn.Sequential(
            nn.Linear(in_dim, bottleneck_dim),
            nn.ReLU(),
            nn.Linear(bottleneck_dim, out_dim)
        )
        
        # 주입된 원본 레이어 고정
        for param in self.original_layer.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.original_layer(x) + self.adapter(x)

# 2. 베이스 모델 생성
model = nn.Sequential(OrderedDict([
    ('input', nn.Linear(1, 16)),
    ('relu1', nn.ReLU()),
    ('dense', nn.Linear(16, 16)),   
    ('relu2', nn.ReLU()),
    ('output', nn.Linear(16, 1))
]))

print("Step 1. 주입 전 원본 모델 구조")
print(model)

for param in model.parameters():
    param.requires_grad = False

# 3. 인젝션
def inject_adapter_by_name(model, target_name, bottleneck_dim=4):
    for name, module in model.named_modules():
        if target_name in name and isinstance(module, nn.Linear):
            
            # 경로 분리
            parts = name.split('.')
            child_name = parts[-1]
            
            # 부모 객체 추적
            parent = model
            for p in parts[:-1]:
                parent = getattr(parent, p)
                
            # setattr을 이용한 레이어 교체 
            setattr(parent, child_name, ResidualAdapter(module, bottleneck_dim))
            print(f"\n'{name}' 레이어를 어댑터로 교체.")

inject_adapter_by_name(model, target_name='dense', bottleneck_dim=4)

print("\nStep 2. 주입 후 모델 파라미터 상태")
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"학습 대상 (Trainable): {name}")

# 4. 데이터셋 및 학습 루프
import torch
X_train = torch.randn(200, 1) * 5 
y_train = torch.cos(X_train) + (torch.randn(200, 1) * 0.1)

# 학습기 세팅 (requires_grad=True인 어댑터 파라미터만 학습)
trainable_params = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.Adam(trainable_params, lr=0.05)
criterion = nn.MSELoss()

print("\nStep 3. 어댑터 학습 시작")
epochs = 5000
for epoch in range(1, epochs + 1):
    predictions = model(X_train)
    loss = criterion(predictions, y_train)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 20 == 0:
        print(f"Epoch [{epoch:3d}/{epochs}] | Loss: {loss.item():.4f}")

print("\nStep 4. 학습 완료")

import matplotlib.pyplot as plt

# 5. 학습 결과 시각화 (Actual vs Predicted)
print("\nStep 5. 학습 결과 시각화 (Matplotlib)")
model.eval()

with torch.no_grad():
    predicted_y = model(X_train).numpy()

X_numpy = X_train.numpy()
y_numpy = y_train.numpy()

plt.figure(figsize=(10, 6))
plt.scatter(X_numpy, y_numpy, color='blue', alpha=0.5, label='Actual Data (Target)')
plt.scatter(X_numpy, predicted_y, color='red', alpha=0.5, label='Adapter Prediction')
plt.title('Adapter Fine-Tuning Result', fontsize=15)
plt.xlabel('X (Input)', fontsize=12)
plt.ylabel('y (Output)', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()