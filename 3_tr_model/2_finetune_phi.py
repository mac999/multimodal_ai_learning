import os
os.environ["WANDB_DISABLED"] = "true"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

# 1. 환경 및 설정
MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
cur_module_folder = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(cur_module_folder, "phi3-rag-finetuned-final")
MAX_SEQ_LENGTH = 1024 # 2048로 설정 시 OOM 발생, 1024로 줄여서 학습 진행 (메모리 상황에 따라 조정 가능)
device = "cuda" if torch.cuda.is_available() else "cpu"
use_bf16 = torch.cuda.is_bf16_supported() if device == "cuda" else False

# 2. 모델 및 토크나이저 로드 (기본 모델)
import bitsandbytes
BNB_AVAILABLE = True

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16 if use_bf16 else torch.float16,
) if BNB_AVAILABLE else None

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, quantization_config=bnb_config, device_map="auto" if device == "cuda" else None, trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    base_model.config.pad_token_id = tokenizer.eos_token_id

# 3. 테스트 프롬프트 및 추론 헬퍼 함수. context와 question 입력해 답변 생성
test_prompts = [
    {
        "context": "The Eiffel Tower was built between 1887-1889 for the 1889 World's Fair in Paris, France. It stands 330 meters tall.",
        "question": "When was the Eiffel Tower built?"
    },
    {
        "context": "Photosynthesis is the process by which plants use sunlight to synthesize nutrients from CO2 and water, producing oxygen.",
        "question": "What does photosynthesis produce?"
    },
    {
        "context": "Machine learning is a subset of AI that enables systems to learn from experience without being explicitly programmed.",
        "question": "What is machine learning?"
    }
]

def run_inference(model_to_test, prompts):
    model_to_test.eval()
    for i, p in enumerate(prompts, 1):
        prompt_text = f"Context: {p['context']}\n\nQuestion: {p['question']}\n\nAnswer:"
        inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model_to_test.generate(**inputs, max_new_tokens=50, pad_token_id=tokenizer.eos_token_id, use_cache=False)
        
        input_len = inputs["input_ids"].shape[1]
        raw_answer = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()
        answer = raw_answer.split("\n\n")[0]
        print(f"Test {i}: {p['question']}\nAnswer: {answer}\n")

# 4. 학습 전(Before) 추론 테스트
print("* [BEFORE TRAINING] INFERENCE\n")
run_inference(base_model, test_prompts)


# 5. 데이터셋 준비 및 포맷팅 (SQuAD v2)
dataset = load_dataset("squad_v2", split="train").filter(lambda x: len(x["answers"]["text"]) > 0)
dataset = dataset.shuffle(seed=42).select(range(11000))

def format_dataset(example):
    if len(example["answers"]["text"]) > 0:
        ans = example["answers"]["text"][0]
    else:
        ans = ""
    user_prompt = f"<|user|>Context: {example['context']}\n\nQuestion: {example['question']}<|end|>"
    assistant_prompt = f"<|assistant|>{ans}<|end|>"
    output = {"text": user_prompt + "\n" + assistant_prompt}
    return output

formatted_ds = dataset.map(format_dataset, remove_columns=dataset.column_names)
train_ds, eval_ds = formatted_ds.select(range(10000)), formatted_ds.select(range(10000, 11000))

# 6. LoRA 어댑터 설정
base_model = prepare_model_for_kbit_training(base_model)

lora_config = LoraConfig(
    task_type="CAUSAL_LM",      # "CAUSAL_LM", "SEQ_2_SEQ_LM", "TOKEN_CLASSIFICATION", "SEQ_CLASSIFICATION", "QUESTION_ANSWERING"
    r=32,                       # rank. 8, 16, 32, 64
    lora_alpha=64,              # scale. r x 2 standard. 
    lora_dropout=0.05,          # LoRA adapter dropout
    bias="none",                # bias training. "none", "all", "lora_only" 
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"] # LoRA adaptation target modules in transformer.
)
peft_model = get_peft_model(base_model, lora_config)
peft_model.gradient_checkpointing_enable()

training_args = SFTConfig(
    output_dir=os.path.join(cur_module_folder, "checkpoints"),
    num_train_epochs=1,
    per_device_train_batch_size=6,  # 8GB VRAM 기준, 4bit/LoRA 사용시 6~8 가능
    gradient_accumulation_steps=2,  # effective batch = 12
    learning_rate=1e-4,
    lr_scheduler_type="cosine",     # linear, constant 등
    optim="adamw_8bit",             # 8bit compression for optimizer states 
    fp16=not use_bf16 and device == "cuda",     
    bf16=use_bf16,                  # bf16 (brain float 16. develop by Google. FP16보다 안정적)
    logging_strategy="steps",       # 옵션은 "no", "epoch", "steps"
    logging_steps=10,               # 10 스텝마다 progress bar에 loss 출력
    save_strategy="no",             # 옵션은 "no", "epoch", "steps"
    report_to="none",               # WandB 등 외부 툴 연동 시 "wandb"로 설정
    dataloader_pin_memory=False,    # Windows에서 안정성 위해 False로 설정. 가속 옵션.
    seed=42,                        
    dataset_text_field="text",      # 데이터셋에서 텍스트가 담긴 필드 이름
    max_seq_length=MAX_SEQ_LENGTH,  # 모델 입력 시 최대 시퀀스 길이
)

# train_ds, eval_ds는 아래에서 명확히 분리되어 사용됨
trainer = SFTTrainer(
    model=peft_model,
    tokenizer=tokenizer,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    args=training_args,
)

print("Starting training...")
trainer.train()

# 8. 파인튜닝 결과 모델(Adapter) 저장
print(f"Saving finetuned model to {SAVE_DIR}...")
trainer.model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)

del peft_model
del trainer
torch.cuda.empty_cache()

# 9. 저장된 모델 로드 (Base Model + Saved LoRA Adapter)
print(f"Loading saved model from {SAVE_DIR} for inference...")
finetuned_model = PeftModel.from_pretrained(base_model, SAVE_DIR)

# 10. 학습 후(After) 추론 테스트
print("* [AFTER TRAINING] INFERENCE\n")
run_inference(finetuned_model, test_prompts)