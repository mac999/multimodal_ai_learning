import torch, os
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import FashionMNIST
from torchvision import transforms
from PIL import Image

import timm
from transformers import DistilBertTokenizer, DistilBertModel
import numpy as np
from tqdm import tqdm

# 1. 설정 및 하이퍼파라미터  
IMAGE_MODEL_NAME = 'resnet18'
TEXT_MODEL_NAME = 'distilbert-base-uncased'
IMAGE_EMBEDDING_DIM = 512   # resnet18의 출력 차원
TEXT_EMBEDDING_DIM = 768    # distilbert의 출력 차원
PROJECTION_DIM = 256        # 최종 임베딩 차원

# 학습 관련 설정
BATCH_SIZE = 64  # 128 -> 64로 감소 (더 안정적인 대조 학습)
NUM_EPOCHS = 10
LEARNING_RATE = 3e-4  # 1e-4 -> 3e-4로 증가 (더 빠른 수렴)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VAL_SPLIT = 0.2  # Validation 데이터 비율

# 텍스트 토큰화 관련 설정
MAX_LENGTH = 32

print(f"Using device: {DEVICE}")

# 2. Fashion-MNIST 데이터셋 준비
cur_module_folder = os.path.dirname(os.path.abspath(__file__))
class FashionMNISTDataset(Dataset):
	"""Fashion-MNIST 데이터셋을 CLIP 학습에 맞게 변환하는 클래스."""
	def __init__(self, train=True, transform=None):
		# torchvision을 통해 Fashion-MNIST 데이터셋을 로드.
		self.dataset = FashionMNIST(root=os.path.join(cur_module_folder, "data"), train=train, download=True) 
		self.transform = transform
		self.tokenizer = DistilBertTokenizer.from_pretrained(TEXT_MODEL_NAME)

		# 숫자 레이블을 실제 클래스 이름으로 매핑.
		self.class_names = [
			"T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
			"Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
		]

	def __len__(self):
		return len(self.dataset)

	def __getitem__(self, idx):
		image, label_idx = self.dataset[idx] # 이미지와 숫자 레이블을 가져옴.
		if self.transform:
			image = self.transform(image) # 이미지 변환 적용
		
		caption = f"a photo of a {self.class_names[label_idx]}" # 숫자 레이블을 이용해 동적으로 캡션을 생성. 실제로는 더 다양한 캡션, 설명을 학습 데이터로 사용할 수 있음. 간단한 예시를 위해 이렇게 작성.
		encoded_caption = self.tokenizer(
			caption, padding='max_length', truncation=True, max_length=MAX_LENGTH, return_tensors='pt'
		) # 캡션을 토큰화.
		
		return {
			'image': image,
			'input_ids': encoded_caption['input_ids'].squeeze(0),
			'attention_mask': encoded_caption['attention_mask'].squeeze(0)
		} # 최종 데이터를 딕셔너리 형태로 반환.

def get_transforms():
	"""이미지 전처리 파이프라인을 정의."""
	return transforms.Compose([
		transforms.Resize((224, 224)),
		transforms.Grayscale(num_output_channels=3), # 흑백 -> 3채널 RGB로 변환
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
	])

# 3. 모델 아키텍처 구현
class ImageEncoder(nn.Module):
	"""이미지 인코더: ResNet18"""
	def __init__(self, pretrained=True, trainable=True):
		super().__init__()
		self.model = timm.create_model(IMAGE_MODEL_NAME, pretrained, num_classes=0, global_pool='avg')
		for p in self.model.parameters():
			p.requires_grad = trainable

	def forward(self, x):
		return self.model(x)

class TextEncoder(nn.Module):
	"""텍스트 인코더: DistilBERT"""
	def __init__(self, pretrained=True, trainable=True):
		super().__init__()
		self.model = DistilBertModel.from_pretrained(TEXT_MODEL_NAME)
		for p in self.model.parameters():
			p.requires_grad = trainable

	def forward(self, input_ids, attention_mask):
		output = self.model(input_ids=input_ids, attention_mask=attention_mask)
		return output.last_hidden_state[:, 0, :] # 문장 전체를 요약해 대표하는 [CLS] 토큰(0번째 토큰)의 임베딩을 사용.

class ProjectionHead(nn.Module):
	"""이미지와 텍스트 임베딩을 동일한 차원으로 투영하는 MLP. 서로 다른 모달리티를 같은 공간에 매핑"""
	def __init__(self, embedding_dim, projection_dim, dropout=0.1):
		super().__init__()
		self.projection = nn.Linear(embedding_dim, projection_dim)
		self.gelu = nn.GELU() # GELU: 부드러운 비선형성 추가 (음수값도 일부 통과)
		self.fc = nn.Linear(projection_dim, projection_dim) # 복잡한 패턴 학습
		self.dropout = nn.Dropout(dropout) # 학습 시 뉴런 무작위 제거 (일반화 성능 향상)
		self.layer_norm = nn.LayerNorm(projection_dim)

	def forward(self, x):
		projected = self.projection(x)
		x = self.gelu(projected)
		x = self.fc(x)
		x = self.dropout(x)
		x = x + projected # 기울기 소실 방지: 깊은 네트워크에서도 역전파가 원활
		x = self.layer_norm(x)
		return x

class CLIPModel(nn.Module):
	"""모든 구성 요소를 결합한 최종 CLIP 모델"""
	def __init__(self):
		super().__init__()
		self.image_encoder = ImageEncoder()
		self.text_encoder = TextEncoder()
		self.image_projection = ProjectionHead(embedding_dim=IMAGE_EMBEDDING_DIM, projection_dim=PROJECTION_DIM)
		self.text_projection = ProjectionHead(embedding_dim=TEXT_EMBEDDING_DIM, projection_dim=PROJECTION_DIM)
		self.temperature = nn.Parameter(torch.tensor(0.07))  # CLIP 논문 권장값

	def forward(self, batch):
		# 각 인코더와 프로젝션 헤드를 통과시켜 임베딩을 얻음. 
        # 학습 시 사용되면 안되는 토큰은 마스킹해서 알려줌.
		image_features = self.image_encoder(batch['image'])
		# attention_mask는 실제 토큰(패딩이 아닌 부분)만 인코더가 처리하도록 알려줌.
		text_features = self.text_encoder(batch['input_ids'], batch['attention_mask']) 
		
		image_embeddings = self.image_projection(image_features)
		text_embeddings = self.text_projection(text_features)
		
		# L2 정규화. 임베딩 벡터의 크기를 1로 맞춰줌. 수식은 벡터를 자신의 L2 노름(크기)으로 나누는 것과 같음.
		image_embeddings = F.normalize(image_embeddings, p=2, dim=-1)
		text_embeddings = F.normalize(text_embeddings, p=2, dim=-1)
		
		return image_embeddings, text_embeddings

# 4. 손실 함수 및 학습/추론 함수
def contrastive_loss(image_embeddings, text_embeddings, temperature):
	"""대조 학습을 위한 대칭적 손실 함수"""
	logits = (text_embeddings @ image_embeddings.T) / temperature   # cosine 유사도 계산값을 온도값으로 나눠 스케일링 처리
	labels = torch.arange(image_embeddings.shape[0], device=DEVICE)
	loss_i = F.cross_entropy(logits.T, labels)
	loss_t = F.cross_entropy(logits, labels)
	return (loss_i + loss_t) / 2.0 # 이미지->텍스트, 텍스트->이미지 방향 모두 고려한 대칭적 손실

def validate(model, dataloader):
	"""Validation 데이터로 모델 성능을 평가하는 함수"""
	model.eval()
	total_loss = 0.0
	total_correct = 0
	total_samples = 0
	
	with torch.no_grad():
		for batch in dataloader:
			batch = {k: v.to(DEVICE) for k, v in batch.items()}
			image_embeddings, text_embeddings = model(batch)
			loss = contrastive_loss(image_embeddings, text_embeddings, model.temperature)
			
			logits = (image_embeddings @ text_embeddings.T) / model.temperature
			predictions = logits.argmax(dim=1)
			labels = torch.arange(image_embeddings.shape[0], device=DEVICE)
			correct = (predictions == labels).sum().item()
			
			total_loss += loss.item() * image_embeddings.shape[0]
	
	avg_loss = total_loss / total_samples
	return avg_loss

def train_one_epoch(model, dataloader, optimizer, epoch, log_file):
	"""하나의 에폭(epoch) 동안 모델을 학습시키는 함수"""
	model.train()
	batch_losses = []
	batch_accs = []
	
	pbar = tqdm(dataloader, desc="Training", leave=True)
	for batch_idx, batch in enumerate(pbar):
		batch = {k: v.to(DEVICE) for k, v in batch.items()}
		image_embeddings, text_embeddings = model(batch)
		loss = contrastive_loss(image_embeddings, text_embeddings, model.temperature)
		
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		
		# loss 계산과 동일하게 temperature 적용
		logits = (image_embeddings @ text_embeddings.T) / model.temperature
		predictions = logits.argmax(dim=1)
		labels = torch.arange(image_embeddings.shape[0], device=DEVICE)
		correct = (predictions == labels).sum().item()
		
		# 배치별 loss와 accuracy 저장
		batch_loss = loss.item()
		batch_losses.append(batch_loss)
		
		# 로그 파일에 기록
		global_step = epoch * len(dataloader) + batch_idx
		log_file.write(f"{global_step},{batch_loss:.6f}\n")
		
	avg_batch_loss = np.mean(batch_losses)
	print(f"Epoch {epoch+1} - Average Batch Loss: {avg_batch_loss:.4f}")
	
	return batch_losses

def run_inference(model, test_dataset):
	"""학습된 모델로 추론을 실행하여 성능을 확인하는 함수"""
	print("\n학습 완료! 테스트 샘플로 추론을 실행합니다...")
	model.eval()
	
	item = test_dataset[0] # 첫 번째 테스트 이미지 사용
	image = item['image'].unsqueeze(0).to(DEVICE)
	true_label = test_dataset.class_names[test_dataset.dataset[0][1]]
	
	# 추론에 사용된 원본 이미지를 파일로 저장
	original_image = test_dataset.dataset[0][0]  # PIL Image
	original_image.save('clip_inference_test_image.png')
	print(f"추론에 사용된 이미지가 'clip_inference_test_image.png'로 저장되었습니다.")
	
	# 이미지 시각화
	import matplotlib.pyplot as plt
	plt.imshow(original_image, cmap='gray')
	plt.title(f"Test Image (True Label: {true_label})")
	plt.axis('off')
	plt.show()
	
	# 모든 클래스에 대한 텍스트 캡션 준비
	all_captions = [f"a photo of a {c}" for c in test_dataset.class_names]
	print(f"테스트 클래스들: {all_captions}")

	with torch.no_grad():
		# 이미지와 텍스트 임베딩 계산
		image_emb = F.normalize(model.image_projection(model.image_encoder(image)), p=2, dim=-1)
		
		tokenizer = test_dataset.tokenizer
		encoded_captions = tokenizer(all_captions, padding=True, truncation=True, return_tensors="pt")
		text_input = {k: v.to(DEVICE) for k, v in encoded_captions.items()}
		text_embs = F.normalize(
			model.text_projection(model.text_encoder(text_input['input_ids'], text_input['attention_mask'])),
			p=2, dim=-1
		)
	
	# 유사도 계산 및 예측
	similarities = (image_emb @ text_embs.T).squeeze(0)
	prediction_idx = similarities.argmax().item()
	
	print(f"실제 정답: '{true_label}'")
	print(f"모델 예측: '{test_dataset.class_names[prediction_idx]}'")
	print("\n--- 유사도 점수 ---")
	for i, cap in enumerate(all_captions):
		print(f"  '{cap}': {similarities[i]:.4f}")

def evaluate_accuracy(model, test_dataset, batch_size=32, k=5):
    """전체 테스트 데이터셋에 대해 Top-1, Top-k Accuracy를 평가하는 함수"""
    print(f"\n테스트 데이터셋 평가를 시작합니다... (Total samples: {len(test_dataset)})")
    model.eval()
    
    # DataLoader 설정 (배치 단위 처리)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    all_captions = [f"a photo of a {c}" for c in test_dataset.class_names]
    tokenizer = test_dataset.tokenizer
    
    with torch.no_grad():
        encoded_captions = tokenizer(all_captions, padding=True, truncation=True, return_tensors="pt")
        text_input = {key: val.to(DEVICE) for key, val in encoded_captions.items()}
        
        # 전체 클래스의 텍스트 임베딩 (Num_Classes, Embedding_Dim)
        text_embs = F.normalize(
            model.text_projection(model.text_encoder(text_input['input_ids'], text_input['attention_mask'])),
            p=2, dim=-1
        )
        
    top1_correct = 0
    topk_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        # 이미지 배치 평가
        for batch in test_loader:
            images = batch['image'].to(DEVICE)
            labels = batch['label'].to(DEVICE) # 정답 클래스의 인덱스
            
            image_embs = F.normalize(model.image_projection(model.image_encoder(images)), p=2, dim=-1)
            
            # 코사인 유사도 계산 (Batch_Size, Num_Classes)
            similarities = image_embs @ text_embs.T
            
            # Top-k 예측 인덱스 추출
            _, topk_preds = similarities.topk(k, dim=-1)
            
            # Top-1 정확도 누적
            top1_preds = topk_preds[:, 0]
            top1_correct += (top1_preds == labels).sum().item()
            
            # Top-k 정확도 누적
            for i in range(len(labels)):
                if labels[i] in topk_preds[i]:
                    topk_correct += 1
                    
            total_samples += labels.size(0)
            
    # 최종 점수 계산
    top1_acc = (top1_correct / total_samples) * 100
    topk_acc = (topk_correct / total_samples) * 100
    
    print("\n=== 평가 결과 ===")
    print(f"Top-1 Accuracy: {top1_acc:.2f}% ({top1_correct}/{total_samples})")
    print(f"Top-{k} Accuracy: {topk_acc:.2f}% ({topk_correct}/{total_samples})")
    
    return top1_acc, topk_acc

# 5. 메인 실행 블록. 데이터셋과 데이터로더 준비 (Train/Val 분할)
if __name__ == '__main__':
	full_dataset = FashionMNISTDataset(train=True, transform=get_transforms())
	train_size = int((1 - VAL_SPLIT) * len(full_dataset))
	val_size = len(full_dataset) - train_size
	train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

	train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
	val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

	print(f"Train 샘플: {train_size}, Validation 샘플: {val_size}")

	# 모델과 옵티마이저 초기화
	model = CLIPModel().to(DEVICE)
	optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

	# 학습 시작
	all_batch_losses = []
	all_batch_accs = []

	# 로그 파일 생성
	log_file = open('clip_training_log.txt', 'w')
	log_file.write("step,loss\n")  # 헤더

	for epoch in tqdm(range(NUM_EPOCHS), desc="Training Epochs"):
		batch_losses = train_one_epoch(model, train_dataloader, optimizer, epoch, log_file)
		train_loss = sum(batch_losses) / len(batch_losses)
		
		# Validation 평가
		val_loss = validate(model, val_dataloader)
		
		print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
		all_batch_losses.extend(batch_losses)
		
		# 학습률 스케줄러 업데이트
		scheduler.step()

	log_file.close()
	print("\n학습 로그가 'clip_training_log.txt'에 저장되었습니다.")

	# 손실 및 정확도 곡선 시각화
	import matplotlib.pyplot as plt
	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
	ax1.plot(range(1, len(all_batch_losses) + 1), all_batch_losses, alpha=0.6)
	ax1.set_title("Training Loss over Batches")
	ax1.set_xlabel("Batch")
	ax1.set_ylabel("Loss")
	ax1.grid()

	ax2.plot(range(1, len(all_batch_accs) + 1), all_batch_accs, color='green', alpha=0.6)
	ax2.set_title("Training Accuracy over Batches")
	ax2.set_xlabel("Batch")
	ax2.set_ylabel("Accuracy (%)")
	ax2.grid()

	plt.tight_layout()
	plt.savefig('clip_training_curves.png', dpi=150, bbox_inches='tight')
	plt.show()

	# 추론 실행
	test_dataset = FashionMNISTDataset(train=False, transform=get_transforms())
	run_inference(model, test_dataset)

	# 전체 테스트 데이터셋에 대한 Top-1, Top-k Accuracy 평가
	evaluate_accuracy(model, test_dataset, batch_size=64, k=5)