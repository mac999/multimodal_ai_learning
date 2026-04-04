import math
import random
import time
from typing import List, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Utilities
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

SPECIAL_TOKENS = {
    "pad": "<pad>",
    "sos": "<s>",
    "eos": "</s>",
    "unk": "<unk>",
}

def set_seed(seed=SEED):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Simple whitespace tokenizer and vocab
class Tokenizer:
    def __init__(self, min_freq: int = 1):
        self.min_freq = min_freq
        self.stoi: Dict[str, int] = {}
        self.itos: List[str] = []

    def build_vocab(self, texts: List[str]):
        freq: Dict[str, int] = {}
        for t in texts:
            for tok in self.tokenize(t):
                freq[tok] = freq.get(tok, 0) + 1

        # special tokens first
        itos = [
            SPECIAL_TOKENS["pad"],
            SPECIAL_TOKENS["sos"],
            SPECIAL_TOKENS["eos"],
            SPECIAL_TOKENS["unk"],
        ]
        for tok, c in sorted(freq.items(), key=lambda x: (-x[1], x[0])):
            if c >= self.min_freq and tok not in itos:
                itos.append(tok)

        self.itos = itos
        self.stoi = {tok: i for i, tok in enumerate(itos)}

    def tokenize(self, text: str) -> List[str]:
        # very simple tokenizer: lowercase and split by whitespace
        return text.strip().lower().split()

    def encode(self, text: str, add_sos: bool = False, add_eos: bool = False) -> List[int]:
        toks = self.tokenize(text)
        ids = []
        if add_sos:
            ids.append(self.stoi[SPECIAL_TOKENS["sos"]])
        for t in toks:
            ids.append(self.stoi.get(t, self.stoi[SPECIAL_TOKENS["unk"]]))
        if add_eos:
            ids.append(self.stoi[SPECIAL_TOKENS["eos"]])
        return ids

    def decode(self, ids: List[int]) -> str:
        toks = []
        for i in ids:
            if i < 0 or i >= len(self.itos):
                continue
            tok = self.itos[i]
            if tok in (SPECIAL_TOKENS["sos"], SPECIAL_TOKENS["eos"], SPECIAL_TOKENS["pad"]):
                continue
            toks.append(tok)
        return " ".join(toks)

    @property
    def pad_id(self) -> int:
        return self.stoi[SPECIAL_TOKENS["pad"]]

    @property
    def sos_id(self) -> int:
        return self.stoi[SPECIAL_TOKENS["sos"]]

    @property
    def eos_id(self) -> int:
        return self.stoi[SPECIAL_TOKENS["eos"]]

    @property
    def unk_id(self) -> int:
        return self.stoi[SPECIAL_TOKENS["unk"]]

    def __len__(self):
        return len(self.itos)


# Toy English-Korean dataset
def build_toy_enko_pairs() -> List[Tuple[str, str]]:
    pairs = [
        ("hello", "안녕"),
        ("hello world", "안녕 세상"),
        ("good morning", "좋은 아침"),
        ("good night", "잘 자"),
        ("thank you", "고마워"),
        ("thank you very much", "정말 고마워"),
        ("see you later", "나중에 봐"),
        ("how are you", "잘 지내"),
        ("i am fine", "난 괜찮아"),
        ("what is your name", "이름이 뭐야"),
        ("my name is john", "내 이름은 존이야"),
        ("nice to meet you", "만나서 반가워"),
        ("where are you from", "어디서 왔어"),
        ("i am from korea", "나는 한국에서 왔어"),
        ("i am from the united states", "나는 미국에서 왔어"),
        ("i like coffee", "난 커피 좋아해"),
        ("i like tea", "난 차 좋아해"),
        ("do you speak english", "영어 할 줄 알아"),
        ("a little", "조금"),
        ("please", "부탁해"),
        ("sorry", "미안해"),
        ("congratulations", "축하해"),
        ("good luck", "행운을 빌어"),
        ("be careful", "조심해"),
        ("see you tomorrow", "내일 보자"),
        ("what time is it", "지금 몇 시야"),
        ("it's okay", "괜찮아"),
        ("i don't know", "모르겠어"),
        ("i understand", "이해했어"),
        ("i don't understand", "이해 못 했어"),
        ("can you help me", "도와줄 수 있어"),
        ("where is the restroom", "화장실이 어디야"),
        ("how much is this", "이거 얼마야"),
        ("too expensive", "너무 비싸"),
        ("i'm hungry", "배고파"),
        ("i'm thirsty", "목말라"),
        ("i'm tired", "피곤해"),
        ("i'm happy", "행복해"),
        ("i'm sad", "슬퍼"),
        ("i'm busy", "바빠"),
        ("i'm studying", "공부 중이야"),
        ("let's go", "가자"),
        ("come here", "여기 와"),
        ("go away", "저리 가"),
        ("wait a moment", "잠깐만"),
        ("one two three", "하나 둘 셋"),
        ("monday tuesday wednesday", "월요일 화요일 수요일"),
        ("i love you", "사랑해"),
        ("i miss you", "보고 싶어"),
        ("see you soon", "곧 보자"),
    ]
    # Duplicate and shuffle a bit to have > 100 samples
    pairs = pairs * 4  # ~200
    random.shuffle(pairs)
    return pairs

class Seq2SeqDataset(Dataset):
    def __init__(self, pairs: List[Tuple[str, str]], src_tok: Tokenizer, tgt_tok: Tokenizer, max_len: int = 40):
        self.data = pairs
        self.src_tok = src_tok
        self.tgt_tok = tgt_tok
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src_txt, tgt_txt = self.data[idx]
        src_ids = self.src_tok.encode(src_txt, add_sos=False, add_eos=True)[: self.max_len]
        tgt_ids = self.tgt_tok.encode(tgt_txt, add_sos=True, add_eos=True)[: self.max_len]
        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(tgt_ids, dtype=torch.long)

def pad_sequence(seqs: List[torch.Tensor], pad_value: int):
    max_len = max(s.size(0) for s in seqs)
    out = torch.full((len(seqs), max_len), pad_value, dtype=torch.long)
    for i, s in enumerate(seqs):
        out[i, : s.size(0)] = s
    return out

def collate_fn(batch, src_pad_id, tgt_pad_id):
    src_seqs, tgt_seqs = zip(*batch)
    src = pad_sequence(list(src_seqs), src_pad_id)
    tgt = pad_sequence(list(tgt_seqs), tgt_pad_id)
    return src, tgt


# Transformer components
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, T, D)
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)

class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V, attn_mask=None):
        # Q,K,V: (B, H, T_q, D_h) / (B, H, T_k, D_h)
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)  # (B,H,Tq,Tk)
        if attn_mask is not None:
            # mask True means mask-out
            scores = scores.masked_fill(attn_mask, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        context = torch.matmul(attn, V)  # (B,H,Tq,Dh)
        return context, attn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.attn = ScaledDotProductAttention(dropout=dropout)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x_q, x_kv, attn_mask=None):
        # x_q: (B,Tq,D), x_kv: (B,Tk,D), attn_mask: (B,1,Tq,Tk) or broadcastable
        residual = x_q

        B, Tq, _ = x_q.size()
        Tk = x_kv.size(1)

        Q = self.W_q(x_q).view(B, Tq, self.num_heads, self.d_k).transpose(1, 2)  # (B,H,Tq,Dh)
        K = self.W_k(x_kv).view(B, Tk, self.num_heads, self.d_k).transpose(1, 2)  # (B,H,Tk,Dh)
        V = self.W_v(x_kv).view(B, Tk, self.num_heads, self.d_k).transpose(1, 2)  # (B,H,Tk,Dh)

        if attn_mask is not None:
            # Expect mask shape (B,1,Tq,Tk) or (1,1,Tq,Tk)
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
            elif attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(1)

        context, _ = self.attn(Q, K, V, attn_mask=attn_mask)
        context = context.transpose(1, 2).contiguous().view(B, Tq, self.d_model)  # (B,Tq,D)
        out = self.out(context)
        out = self.dropout(out)
        out = self.norm(out + residual)
        return out

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        residual = x
        x = self.fc1(x)
        x = F.relu(x, inplace=True)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.norm(x + residual)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ff = PositionwiseFeedForward(d_model, d_ff, dropout)

    def forward(self, x, src_pad_mask):
        # src_pad_mask: (B,1,1,T) True for pad
        x = self.self_attn(x, x, attn_mask=src_pad_mask)
        x = self.ff(x)
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ff = PositionwiseFeedForward(d_model, d_ff, dropout)

    def forward(self, x, enc_out, tgt_mask, src_pad_mask):
        # tgt_mask: (B,1,T,T) True for mask
        # src_pad_mask: (B,1,1,S) True for pad
        x = self.self_attn(x, x, attn_mask=tgt_mask)
        # build cross mask from src_pad_mask but with T dimension aligned
        x = self.cross_attn(x, enc_out, attn_mask=src_pad_mask)
        x = self.ff(x)
        return x

class Encoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, num_layers: int, num_heads: int, d_ff: int, dropout: float, max_len: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = PositionalEncoding(d_model, max_len=max_len, dropout=dropout)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])

    def forward(self, src_ids, src_pad_mask):
        # src_ids: (B,S)
        x = self.embed(src_ids)  # (B,S,D)
        x = self.pos(x)
        for layer in self.layers:
            x = layer(x, src_pad_mask)
        return x

class Decoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, num_layers: int, num_heads: int, d_ff: int, dropout: float, max_len: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = PositionalEncoding(d_model, max_len=max_len, dropout=dropout)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tgt_ids, enc_out, tgt_mask, src_pad_mask):
        # tgt_ids: (B,T)
        x = self.embed(tgt_ids)
        x = self.pos(x)
        for layer in self.layers:
            x = layer(x, enc_out, tgt_mask, src_pad_mask)
        x = self.norm(x)
        return x

class Transformer(nn.Module):
    def __init__(self, src_vocab: int, tgt_vocab: int, d_model: int = 256, num_layers: int = 2, num_heads: int = 4, d_ff: int = 512, dropout: float = 0.1, max_len: int = 128, tie_output: bool = True):
        super().__init__()
        self.encoder = Encoder(src_vocab, d_model, num_layers, num_heads, d_ff, dropout, max_len)
        self.decoder = Decoder(tgt_vocab, d_model, num_layers, num_heads, d_ff, dropout, max_len)
        self.out_proj = nn.Linear(d_model, tgt_vocab, bias=False)
        if tie_output:
            # Tie decoder embedding with output projection when sizes match
            self.out_proj.weight = self.decoder.embed.weight

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src_ids, src_pad_mask):
        return self.encoder(src_ids, src_pad_mask)

    def decode(self, tgt_ids, enc_out, tgt_mask, src_pad_mask):
        return self.decoder(tgt_ids, enc_out, tgt_mask, src_pad_mask)

    def forward(self, src_ids, tgt_inp_ids, src_pad_mask, tgt_mask):
        enc_out = self.encode(src_ids, src_pad_mask)
        dec_out = self.decode(tgt_inp_ids, enc_out, tgt_mask, src_pad_mask)
        logits = self.out_proj(dec_out)
        return logits


# Mask helpers
def make_pad_mask(seq: torch.Tensor, pad_id: int):
    # seq: (B,T), return mask shape (B,1,1,T) True for pad
    mask = (seq == pad_id).unsqueeze(1).unsqueeze(1)
    return mask  # (B,1,1,T)

def make_subsequent_mask(size: int):
    # return (1,1,T,T) upper-triangular True to mask future
    mask = torch.triu(torch.ones(size, size, dtype=torch.bool), diagonal=1)
    return mask.unsqueeze(0).unsqueeze(0)

def combine_masks(pad_mask: torch.Tensor, subseq_mask: torch.Tensor, B: int):
    # pad_mask: (B,1,1,S) or (B,1,1,T)
    # subseq_mask: (1,1,T,T)
    # For target self-attn, need shape (B,1,T,T)
    if pad_mask.size(-1) != subseq_mask.size(-1):
        # If pad mask length differs (shouldn't for tgt), adapt
        pad_mask = pad_mask.expand(B, -1, subseq_mask.size(-1), -1)  # Not used here
    mask = subseq_mask.expand(B, -1, -1, -1) | False  # (B,1,T,T)
    return mask

def make_tgt_mask(tgt_inp: torch.Tensor, pad_id: int):
    # Combine padding and subsequent masks
    B, T = tgt_inp.size()
    pad_mask = make_pad_mask(tgt_inp, pad_id)  # (B,1,1,T) for keys
    subseq = make_subsequent_mask(T).to(tgt_inp.device)  # (1,1,T,T)
    # For self-attn in decoder, we need to prevent attending to pad positions as well.
    # Build a mask with shape (B,1,T,T): broadcast pad_mask to (B,1,T,T)
    pad_for_tgt = pad_mask.expand(B, 1, T, T)  # mask keys that are pad across all queries
    tgt_mask = subseq | pad_for_tgt
    return tgt_mask  # (B,1,T,T)


# Training helpers
def greedy_decode(model: Transformer, src_ids: torch.Tensor, src_pad_mask: torch.Tensor, max_len: int, sos_id: int, eos_id: int):
    model.eval()
    B = src_ids.size(0)
    device = src_ids.device
    with torch.no_grad():
        enc_out = model.encode(src_ids, src_pad_mask)
        ys = torch.full((B, 1), sos_id, dtype=torch.long, device=device)
        for _ in range(max_len - 1):
            tgt_mask = make_tgt_mask(ys, pad_id=-1)  # pad_id not used here as ys has no pad yet
            dec_out = model.decode(ys, enc_out, tgt_mask, src_pad_mask)
            logits = model.out_proj(dec_out[:, -1:, :])  # (B,1,V)
            next_token = logits.argmax(dim=-1)  # (B,1)
            ys = torch.cat([ys, next_token], dim=1)
            if (next_token == eos_id).all():
                break
        return ys

class NoamScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, d_model, warmup_steps=4000, last_epoch=-1):
        self.d_model = d_model
        self.warmup = warmup_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = max(self._step_count, 1)
        scale = (self.d_model ** -0.5) * min(step ** -0.5, step * (self.warmup ** -1.5))
        return [base_lr * scale for base_lr in self.base_lrs]


# Main training
#
set_seed()
device = get_device()
print(f"Device: {device}")

# Hyperparameters tuned for ~6GB VRAM
d_model = 256
num_layers = 2
num_heads = 4
d_ff = 512
dropout = 0.1
max_len = 64
batch_size = 32
epochs = 100
lr = 1.0  # Noam uses this as base scale
warmup_steps = 200

# Build dataset
pairs = build_toy_enko_pairs()
# Split train/val
split = int(len(pairs) * 0.9)
train_pairs = pairs[:split]
val_pairs = pairs[split:]

# Build tokenizers from training data only
src_texts = [p[0] for p in train_pairs]
tgt_texts = [p[1] for p in train_pairs]

src_tok = Tokenizer(min_freq=1)
tgt_tok = Tokenizer(min_freq=1)
src_tok.build_vocab(src_texts + [SPECIAL_TOKENS["eos"]])
tgt_tok.build_vocab(tgt_texts + [SPECIAL_TOKENS["eos"]])

print(f"Src vocab: {len(src_tok)}, Tgt vocab: {len(tgt_tok)}")

train_ds = Seq2SeqDataset(train_pairs, src_tok, tgt_tok, max_len=max_len)
val_ds = Seq2SeqDataset(val_pairs, src_tok, tgt_tok, max_len=max_len)

train_loader = DataLoader(
    train_ds, batch_size=batch_size, shuffle=True,
    collate_fn=lambda b: collate_fn(b, src_tok.pad_id, tgt_tok.pad_id),
    drop_last=False
)
val_loader = DataLoader(
    val_ds, batch_size=batch_size, shuffle=False,
    collate_fn=lambda b: collate_fn(b, src_tok.pad_id, tgt_tok.pad_id),
    drop_last=False
)

model = Transformer(
    src_vocab=len(src_tok),
    tgt_vocab=len(tgt_tok),
    d_model=d_model,
    num_layers=num_layers,
    num_heads=num_heads,
    d_ff=d_ff,
    dropout=dropout,
    max_len=max_len,
    tie_output=True
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)
scheduler = NoamScheduler(optimizer, d_model=d_model, warmup_steps=warmup_steps)
criterion = nn.CrossEntropyLoss(ignore_index=tgt_tok.pad_id)

scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

def run_epoch(loader, train=True):
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_tokens = 0
    total_correct = 0
    mode = "Train" if train else "Valid"

    pbar = tqdm(loader, desc=mode)
    for src, tgt in pbar:
        src = src.to(device)  # (B,S)
        tgt = tgt.to(device)  # (B,T) includes <s> and </s>

        # Prepare inputs/targets
        tgt_inp = tgt[:, :-1]  # shift right (starts with <s>)
        tgt_out = tgt[:, 1:]   # predict next tokens (ends with </s>)

        # Masks
        src_pad_mask = make_pad_mask(src, src_tok.pad_id).to(device)     # (B,1,1,S)
        tgt_mask = make_tgt_mask(tgt_inp, tgt_tok.pad_id).to(device)     # (B,1,T,T)

        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            logits = model(src, tgt_inp, src_pad_mask, tgt_mask)  # (B,T,V)
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
        # Accuracy 계산
        predictions = logits.argmax(dim=-1)
        mask = (tgt_out != tgt_tok.pad_id)
        correct = ((predictions == tgt_out) & mask).sum().item()
        # Accuracy 계산
        predictions = logits.argmax(dim=-1)
        mask = (tgt_out != tgt_tok.pad_id)
        correct = ((predictions == tgt_out) & mask).sum().item()
        
        if train:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

        tokens = (tgt_out != tgt_tok.pad_id).sum().item()
        total_loss += loss.item() * max(tokens, 1)
        total_tokens += max(tokens, 1)
        total_correct += correct

    avg_loss = total_loss / max(total_tokens, 1)
    acc = 100 * total_correct / max(total_tokens, 1)
    print(f"{mode}: loss={avg_loss:.4f}, acc={acc:.2f}%")
    return avg_loss

for epoch in range(1, epochs + 1):
    print(f"\nEpoch {epoch}/{epochs}")
    run_epoch(train_loader, train=True)
    with torch.no_grad():
        run_epoch(val_loader, train=False)

# Quick demo translations
model.eval()
examples = [
    "hello",
    "good morning",
    "i am from korea",
    "i like coffee",
    "see you tomorrow",
    "how much is this",
]
print("\nDemo translations:")
for s in examples:
    src_ids = src_tok.encode(s, add_eos=True)
    src_tensor = torch.tensor(src_ids, dtype=torch.long, device=device).unsqueeze(0)
    src_pad_mask = make_pad_mask(src_tensor, src_tok.pad_id)
    out_ids = greedy_decode(model, src_tensor, src_pad_mask.to(device), max_len=max_len, sos_id=tgt_tok.sos_id, eos_id=tgt_tok.eos_id)
    print(f"EN: {s} -> KO: {tgt_tok.decode(out_ids[0].tolist())}")
