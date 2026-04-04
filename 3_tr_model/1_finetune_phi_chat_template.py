from transformers import AutoTokenizer

# 궁금한 모델의 ID를 입력하세요
model_id = "microsoft/Phi-3-mini-4k-instruct" # 또는 "google/gemma-2b-it"
tokenizer = AutoTokenizer.from_pretrained(model_id)

# 토크나이저에 내장된 챗 템플릿 출력
print(tokenizer.chat_template)

# if message['role'] == 'system' %}{{'<|system|>' + message['content'] + '<|end|>
# if message['role'] == 'user' %}{{'<|user|>' + message['content'] + '<|end|>
# if message['role'] == 'assistant' %}{{'<|assistant|>' + message['content'] + '<|end|>