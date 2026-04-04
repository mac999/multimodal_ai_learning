import os
os.environ["WANDB_DISABLED"] = "true"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from datasets import load_dataset
from trl import SFTTrainer

# 1. 환경 및 설정
cur_module_folder = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
SAVE_DIR = os.path.join(cur_module_folder, "phi3-rag-finetuned-final")
MAX_SEQ_LENGTH = 2048
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

# 3. 추론 헬퍼 함수 및 테스트 프롬프트
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
        prompt_text = f"<|user|>Context: {p['context']}\n\nQuestion: {p['question']}<|end|>\n<|assistant|>"
        inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model_to_test.generate(**inputs, max_new_tokens=50, pad_token_id=tokenizer.eos_token_id)
        
        input_len = inputs["input_ids"].shape[1]
        raw_answer = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()
        answer = raw_answer.split("\n\n")[0]
        print(f"Test {i}: {p['question']}\nAnswer: {answer}\n")

# 4. 학습 전(Before) 추론 테스트
print("* [BEFORE TRAINING] INFERENCE\n")
run_inference(base_model, test_prompts)

# 5. 저장된 모델 로드 (Base Model + Saved LoRA Adapter)
print(f"Loading saved model from {SAVE_DIR} for inference...")
finetuned_model = PeftModel.from_pretrained(base_model, SAVE_DIR)

# 6. 학습 후(After) 추론 테스트
print("* [AFTER TRAINING] INFERENCE\n")
run_inference(finetuned_model, test_prompts)