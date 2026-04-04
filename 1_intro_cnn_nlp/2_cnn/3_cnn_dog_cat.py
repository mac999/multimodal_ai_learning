import torchvision.transforms as transforms

# 이미지 변환 규칙을 순서대로 정의
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # 크기 맞추기
    transforms.ToTensor(),        # 텐서 데이터타입으로 변환
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # 밝기/대비 표준화
])

import os
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset

# 파이토치 Dataset 클래스를 상속받아 우리만의 데이터 학습 모델 정의
class CatsVsDogsDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = os.listdir(root_dir) # 데이터 폴더 안의 모든 파일 이름을 리스트로 저장
        
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx): # 특정 인덱스(idx)의 데이터(이미지와 레이블)를 요청받았을 때 실행
        img_name = self.image_files[idx] 				 # idx에 해당하는 파일 이름을 획득
        img_path = os.path.join(self.root_dir, img_name) # 파일 이름과 폴더 경로를 합쳐 전체 파일 경로를 생성
        image = Image.open(img_path).convert('RGB') 	 # 'PIL' 라이브러리로 이미지를 열고, RGB 형식으로 통일
        
        label = 0 if 'cat' in img_name else 1   # 파일 이름에 'cat'이 있으면 레이블을 0, 아니면(dog이면) 1로 설정

        if self.transform: # 변환 규칙(transform)이 있다면, 이미지를 규칙에 맞게 가공
            image = self.transform(image)
            
        return image, label  # 가공된 이미지와 정답(레이블)을 함께 반환

from torch.utils.data import DataLoader, random_split

module_path = os.path.dirname(os.path.abspath(__file__)) # 전체 데이터를 불러오기
train_data_path = os.path.join(module_path, 'dogs-vs-cats', 'train')
full_dataset = CatsVsDogsDataset(root_dir=train_data_path, transform=transform)

train_size = int(0.8 * len(full_dataset)) # 전체 데이터를 훈련용(80%)과 검증용(20%)으로 분할
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

batch_size = 1024  # 데이터 배치 크기
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) # 학습 데이터를 무작위로 섞어(shuffle=True) 모델이 패턴을 외우지 못하게 함
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)  # 검증 데이터는 순서대로 데이터를 제공

print(f"훈련 데이터: {len(train_dataset)}개")
print(f"검증 데이터: {len(val_dataset)}개")

import torch.nn as nn
import torch.nn.functional as F

class SimpleResCNN(nn.Module):
    def __init__(self):
        super(SimpleResCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(16, 16, 3, padding=1)  # 첫 번째 conv (채널 유지)
        self.conv3 = nn.Conv2d(16, 16, 3, padding=1)  # 두 번째 conv (채널 유지)
        
        self.fc1 = nn.Linear(16 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, 2) # Classification layers

    def forward(self, x):
        x = F.relu(self.conv1(x))  	# (3,64,64) → (16,64,64)
        x = self.pool(x)           	# (16,64,64) → (16,32,32)
        skip_connect = x  			# skip connection을 위해 입력 저장
        out = self.conv2(x)
        out = F.relu(out)    		# conv + relu
        out = self.conv3(out)       
        out = out + skip_connect  	# Skip connection: F(x) + x
        out = F.relu(out)         	# 최종 activation
        x = self.pool(out)  # (16,32,32) → (16,16,16)
        x = x.view(-1, 16 * 16 * 16)
        x = self.fc1(x)
        x = F.relu(x) # 분류를 위한 pooling 및 FC layer
        x = self.fc2(x)
        return x

model = SimpleResCNN()
print(model)

import torch.optim as optim, numpy as np, torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"사용 중인 디바이스: {device}")
if torch.cuda.is_available():	
    print(f"GPU 이름: {torch.cuda.get_device_name(0)}")
    print(f"GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

model = model.to(device) # 모델을 GPU로 이동

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 20 			# 에폭 횟수 설정
best_val_loss = np.inf 		# 최고 점수를 기록할 변수를 아주 큰 값으로 초기화

model_fname = '' # 모델 저장 파일 이름을 저장할 변수 초기화
print("훈련과 검증을 시작합니다...")
for epoch in tqdm(range(num_epochs), desc="Epoch"): # 정해진 횟수만큼 전체 데이터셋을 반복 학습

    model.train() # 모델을 '학습 모드'로 전환
    running_loss = 0.0
    train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} Training")
    for inputs, labels in train_pbar:
        inputs, labels = inputs.to(device), labels.to(device)  # 데이터를 GPU로 이동
        optimizer.zero_grad()      
        outputs = model(inputs)    
        loss = criterion(outputs, labels) 
        loss.backward()            
        optimizer.step()           
        running_loss += loss.item()
        
        train_pbar.set_postfix({'loss': f'{loss.item():.4f}', 'avg_loss': f'{running_loss/(train_pbar.n+1):.4f}'})

    print(f'[{epoch + 1}] 훈련 손실: {running_loss / len(train_loader):.3f}')

    model.eval() # 모델을 '평가 모드'로 전환 (학습 기능 일시 정지)
    val_loss = 0.0
    val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1} Validation")
    with torch.no_grad(): # 정답을 봐도 학습(가중치 업데이트)하지 않도록 설정
        for inputs, labels in val_pbar:
            inputs, labels = inputs.to(device), labels.to(device)  # 데이터를 GPU로 이동
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            val_pbar.set_postfix({'loss': f'{loss.item():.4f}', 'avg_loss': f'{val_loss/(val_pbar.n+1):.4f}'})
    
    current_val_loss = val_loss / len(val_loader)
    print(f'[{epoch + 1}] 검증 손실: {current_val_loss:.3f}')

    if current_val_loss < best_val_loss: # 최고 점수 모델 저장
        best_val_loss = current_val_loss

        model_fname = os.path.join(module_path, 'best_cnn_model.pth') # 모델 저장 파일 이름 설정
        torch.save(model.state_dict(), model_fname) # 현재 모델의 상태 저장
        print(f'모델을 "{model_fname}"에 저장했습니다')

print('훈련 종료!')

best_model = SimpleResCNN().to(device)  # 모델을 GPU로 이동
best_model.load_state_dict(torch.load(model_fname))
best_model.eval() # 평가 모드로 설정

image_path = os.path.join(module_path, 'new_image.jpg')  # 절대 경로로 변경
image = Image.open(image_path).convert('RGB')
input_tensor = transform(image) # 훈련 때와 똑같은 변환 규칙(transform)을 적용
input_batch = input_tensor.unsqueeze(0).to(device)  # 모델은 이미지 묶음(배치) 단위로 입력을 받으므로, 한 장의 이미지를 묶음으로 변환

classes = ('cat', 'dog') # 정답 종류를 정의(0번째는 'cat', 1번째는 'dog')
with torch.no_grad(): 
    output = best_model(input_batch)

_, predicted_idx = torch.max(output, 1) # 모델의 출력값 중 가장 높은 점수를 받은 인덱스를 획득
predicted_class = classes[predicted_idx.item()] # 해당 인덱스에 맞는 클래스 이름을 가져옴

print(f'\n최종 예측 결과: "{image_path}" 이미지는 "{predicted_class}" 입니다.')