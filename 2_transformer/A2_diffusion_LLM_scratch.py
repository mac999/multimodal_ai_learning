import math
import random
import time
import os
from typing import List, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# 1. 유틸리티 및 설정 
SEED = 42
def set_seed(seed=SEED):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

SPECIAL_TOKENS = {
    "pad": "<pad>",
    "sos": "<s>",
    "eos": "</s>",
    "unk": "<unk>",
    "mask": "<mask>",
}

# 2. 토크나이저 
class Tokenizer:
    def __init__(self, min_freq: int = 1):
        self.min_freq = min_freq
        self.stoi: Dict[str, int] = {}
        self.itos: List[str] = []

    def build_vocab(self, texts: List[str]):
        freq: Dict[str, int] = {}
        for txt in texts:
            for tok in self.tokenize(txt):
                freq[tok] = freq.get(tok, 0) + 1
        
        itos = list(SPECIAL_TOKENS.values())
        for tok, c in sorted(freq.items(), key=lambda x: (-x[1], x[0])):
            if c >= self.min_freq and tok not in itos:
                itos.append(tok)

        self.itos = itos
        self.stoi = {tok: i for i, tok in enumerate(itos)}

    def tokenize(self, text: str) -> List[str]:
        return text.strip().lower().split()

    def encode(self, text: str, add_sos: bool = False, add_eos: bool = False) -> List[int]:
        toks = self.tokenize(text)
        ids = []
        if add_sos: ids.append(self.sos_id)
        ids.extend(self.stoi.get(t, self.unk_id) for t in toks)
        if add_eos: ids.append(self.eos_id)
        return ids

    def decode(self, ids: List[int]) -> str:
        toks = [self.itos[i] for i in ids if i < len(self.itos) and self.itos[i] not in [SPECIAL_TOKENS["pad"], SPECIAL_TOKENS["sos"], SPECIAL_TOKENS["eos"]]]
        return " ".join(toks)

    @property
    def pad_id(self) -> int: return self.stoi[SPECIAL_TOKENS["pad"]]
    @property
    def sos_id(self) -> int: return self.stoi[SPECIAL_TOKENS["sos"]]
    @property
    def eos_id(self) -> int: return self.stoi[SPECIAL_TOKENS["eos"]]
    @property
    def unk_id(self) -> int: return self.stoi[SPECIAL_TOKENS["unk"]]
    @property
    def mask_id(self) -> int: return self.stoi[SPECIAL_TOKENS["mask"]]
    def __len__(self): return len(self.itos)

# 3. 데이터셋 및 데이터로더  
def build_toy_enko_pairs() -> List[Tuple[str, str]]:
    pairs = [
        ("hello", "안녕"), ("hello world", "안녕 세상"), ("good morning", "좋은 아침"),
        ("good night", "잘 자"), ("thank you", "고마워"), ("thank you very much", "정말 고마워"),
        ("see you later", "나중에 봐"), ("how are you", "잘 지내"), ("i am fine", "난 괜찮아"),
        ("what is your name", "이름이 뭐야"), ("my name is john", "내 이름은 존이야"),
        ("nice to meet you", "만나서 반가워"), ("where are you from", "어디서 왔어"),
        ("i am from korea", "나는 한국에서 왔어"), ("i am from the united states", "나는 미국에서 왔어"),
        ("i like coffee", "난 커피 좋아해"), ("i like tea", "난 차 좋아해"),
    ]
    return (pairs * 10)

class MaskedSeq2SeqDataset(Dataset):
    def __init__(self, pairs: List[Tuple[str, str]], src_tok: Tokenizer, tgt_tok: Tokenizer, max_len: int = 40):
        self.pairs = pairs
        self.src_tok, self.tgt_tok = src_tok, tgt_tok
        self.max_len = max_len

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src_txt, tgt_txt = self.pairs[idx]
        src_ids = self.src_tok.encode(src_txt, add_eos=True)[:self.max_len]
        tgt_ids = self.tgt_tok.encode(tgt_txt, add_sos=True, add_eos=True)[:self.max_len]
        
        # 코사인 스케줄에 따라 마스킹할 개수 결정 (개념적인 시간 t에 해당)
        t_rand = random.random()
        num_to_mask = math.ceil(len(tgt_ids) * math.cos(t_rand * math.pi / 2))
        
        masked_tgt_ids = list(tgt_ids)
        maskable_indices = [i for i, t_id in enumerate(tgt_ids) if t_id not in (self.tgt_tok.sos_id, self.tgt_tok.eos_id)]
        
        if len(maskable_indices) > 0 and num_to_mask > 0:
            indices_to_mask = random.sample(maskable_indices, min(num_to_mask, len(maskable_indices)))
            for i in indices_to_mask:
                masked_tgt_ids[i] = self.tgt_tok.mask_id

        return (torch.tensor(src_ids, dtype=torch.long),
                torch.tensor(masked_tgt_ids, dtype=torch.long),
                torch.tensor(tgt_ids, dtype=torch.long))

def pad_sequence(seqs: List[torch.Tensor], pad_value: int, max_len=None):
    L = max(s.size(0) for s in seqs) if max_len is None else max_len
    out = torch.full((len(seqs), L), pad_value, dtype=torch.long)
    for i, s in enumerate(seqs):
        out[i, :s.size(0)] = s
    return out

def collate_fn(batch, src_pad_id, tgt_pad_id):
    src, masked_tgt, tgt = zip(*batch)
    src_padded = pad_sequence(list(src), src_pad_id)
    masked_tgt_padded = pad_sequence(list(masked_tgt), tgt_pad_id)
    tgt_padded = pad_sequence(list(tgt), tgt_pad_id, max_len=masked_tgt_padded.size(1))
    return src_padded, masked_tgt_padded, tgt_padded

# 4. 모델 아키텍처 
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout)
    def forward(self, x):
        return self.dropout(x + self.pe[:x.size(0)])

class NARTransformer(nn.Module):
    def __init__(self, src_vocab: int, tgt_vocab: int, d_model: int = 256, nhead: int = 4, num_enc_layers: int = 2, num_dec_layers: int = 2, dim_ff: int = 512, dropout: float = 0.1, max_len: int = 128):
        super().__init__()
        self.d_model = d_model
        self.src_embedding = nn.Embedding(src_vocab, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len, dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_ff, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_enc_layers)
        
        decoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_ff, dropout, batch_first=True)
        self.decoder = nn.TransformerEncoder(decoder_layer, num_dec_layers)
        
        self.out_proj = nn.Linear(d_model, tgt_vocab)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, tgt, src_pad_mask, tgt_pad_mask):
        src_embed = self.pos_encoder(self.src_embedding(src).transpose(0, 1)).transpose(0, 1)
        tgt_embed = self.pos_encoder(self.tgt_embedding(tgt).transpose(0, 1)).transpose(0, 1)
        memory = self.encoder(src_embed, src_key_padding_mask=src_pad_mask)
        output = self.decoder(tgt_embed, src_key_padding_mask=tgt_pad_mask)
        combined = output + memory.mean(dim=1, keepdim=True)
        return self.out_proj(combined)

    def encode(self, src, src_pad_mask):
        src_embed = self.pos_encoder(self.src_embedding(src).transpose(0, 1)).transpose(0, 1)
        return self.encoder(src_embed, src_key_padding_mask=src_pad_mask)
    
    def decode_step(self, tgt, memory, tgt_pad_mask):
        tgt_embed = self.pos_encoder(self.tgt_embedding(tgt).transpose(0, 1)).transpose(0, 1)
        output = self.decoder(tgt_embed, src_key_padding_mask=tgt_pad_mask)
        combined = output + memory.mean(dim=1, keepdim=True)
        return self.out_proj(combined)

# 5. 추론 함수 (Fast-dLLM)
@torch.no_grad()
def fast_dllm_decode(
    model: NARTransformer,
    src_tensor: torch.Tensor,
    src_text: str,
    tgt_tok: Tokenizer,
    max_len: int,
    num_blocks: int = 2,
    steps_per_block: int = 12,
    confidence_threshold: float = 0.9,
    visualize: bool = False,
):
    model.eval()
    device = src_tensor.device
    B = src_tensor.size(0)

    def _visual_print(step_str, current_ids, new_ids_mask):
        GREEN = '\033[92m'
        DIM = '\033[2m'
        RESET = '\033[0m'
        
        toks = []
        for i, token_id in enumerate(current_ids):
            tok = tgt_tok.itos[token_id]
            if tok in [SPECIAL_TOKENS["pad"], SPECIAL_TOKENS["sos"], SPECIAL_TOKENS["eos"]]:
                continue
            
            if new_ids_mask[i]:
                toks.append(f"{GREEN}{tok}{RESET}")
            elif token_id == tgt_tok.mask_id:
                toks.append(f"{DIM}█{RESET}")
            else:
                toks.append(tok)
        
        os.system('cls' if os.name == 'nt' else 'clear')
        print("Fast-dLLM Decoding")
        print(f"EN: {src_text}")
        print(f"KO: {' '.join(toks)}")
        print(f"\n{step_str}")
        time.sleep(0.15)

    src_pad_mask = (src_tensor == tgt_tok.pad_id)
    memory = model.encode(src_tensor, src_pad_mask)

    ys = torch.full((B, max_len), tgt_tok.mask_id, dtype=torch.long, device=device)
    ys[:, 0] = tgt_tok.sos_id
    
    if visualize:
        _visual_print("Step 0: Initial Mask", ys[0], torch.zeros_like(ys[0], dtype=torch.bool))

    block_size = math.ceil((max_len - 1) / num_blocks)
    total_steps = steps_per_block * num_blocks
    step_count = 0

    for k in range(num_blocks):
        start_idx = 1 + k * block_size
        end_idx = min(start_idx + block_size, max_len)
        
        # 수정된 부분 시작
        # 아래 루프가 논문의 개념적인 '시간 t'의 흐름을 나타냅니다.
        for t in range(steps_per_block):
        # 수정된 부분 끝
            step_count += 1
            
            tgt_pad_mask = (ys == tgt_tok.pad_id)
            logits = model.decode_step(ys, memory, tgt_pad_mask)
            probs = F.softmax(logits, dim=-1)
            confidences, predictions = probs.max(dim=-1)
            
            current_block_mask = torch.zeros_like(ys, dtype=torch.bool)
            current_block_mask[:, start_idx:end_idx] = True
            mask_positions = (ys == tgt_tok.mask_id) & current_block_mask
            
            if not mask_positions.any(): break
            
            unmask_candidates = (confidences > confidence_threshold) & mask_positions
            
            if not unmask_candidates.any():
                masked_confidences = confidences.where(mask_positions, torch.tensor(-1.0, device=device))
                if masked_confidences.max() > -1:
                    highest_idx = masked_confidences.argmax(dim=1, keepdim=True)
                    unmask_candidates.scatter_(1, highest_idx, 1)

            if visualize:
                _visual_print(f"Step {step_count}/{total_steps} (Block {k+1}/{num_blocks})", ys[0], unmask_candidates[0])

            ys.masked_scatter_(unmask_candidates, predictions[unmask_candidates])

    final_seqs = []
    for i in range(B):
        seq = ys[i].tolist()
        try:
            eos_pos = seq.index(tgt_tok.eos_id)
            seq = seq[:eos_pos]
        except ValueError:
            pass
        final_seqs.append(tgt_tok.decode(seq))

    return final_seqs

# 6. 메인 학습 및 실행 (변경 없음)
set_seed()
device = get_device()
print(f"Using device: {device}")

d_model = 128
nhead = 4
num_layers = 2
dim_ff = 256
dropout = 0.1
max_len = 32
batch_size = 32
epochs = 100
lr = 0.001

pairs = build_toy_enko_pairs()
random.shuffle(pairs)
split = int(len(pairs) * 0.9)
train_pairs, val_pairs = pairs[:split], pairs[split:]

src_tok = Tokenizer()
tgt_tok = Tokenizer()
src_tok.build_vocab([p[0] for p in train_pairs])
tgt_tok.build_vocab([p[1] for p in train_pairs])

train_ds = MaskedSeq2SeqDataset(train_pairs, src_tok, tgt_tok, max_len=max_len)
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=lambda b: collate_fn(b, src_tok.pad_id, tgt_tok.pad_id))

model = NARTransformer(len(src_tok), len(tgt_tok), d_model, nhead, num_layers, num_layers, dim_ff, dropout, max_len).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss(ignore_index=tgt_tok.pad_id)

print("\nStarting Training")
for epoch in range(1, epochs + 1):
    model.train()
    total_loss = 0
    start_time = time.time()
    for src, masked_tgt, tgt in train_loader:
        src, masked_tgt, tgt = src.to(device), masked_tgt.to(device), tgt.to(device)
        
        src_pad_mask = (src == src_tok.pad_id)
        tgt_pad_mask = (masked_tgt == tgt_tok.pad_id)
        
        optimizer.zero_grad()
        logits = model(src, masked_tgt, src_pad_mask, tgt_pad_mask)
        loss = criterion(logits.view(-1, len(tgt_tok)), tgt.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    elapsed = time.time() - start_time
    print(f"Epoch {epoch}/{epochs} | Loss: {avg_loss:.4f} | Time: {elapsed:.2f}s")

examples = ["hello", "thank you very much", "good morning", "i like coffee", "how are you"]
print(f"\nDemo Translations (Visualizing first example)")
for i, ex in enumerate(examples):
    src_tensor = torch.tensor([src_tok.encode(ex, add_eos=True)], dtype=torch.long, device=device)
    
    should_visualize = (i < 2)
    
    translation = fast_dllm_decode(model, src_tensor, ex, tgt_tok, max_len=max_len, visualize=should_visualize)
    
    if not should_visualize:
            print(f"EN: {ex} -> KO: {translation[0]}")
    else:
        time.sleep(1)
        final_display = f"EN: {ex} -> KO: {translation[0]}"
        os.system('cls' if os.name == 'nt' else 'clear')
        print("Fast-dLLM Decoding")
        print(f"{final_display} (Done!)")
        print("\nOther Examples")
