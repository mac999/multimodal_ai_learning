# Model finetuning process for QA Tasks using LoRA and DPO
#
# --- STEP 0: Load Model with Quantization ---
# Load base model in 4-bit to save VRAM (Quantization)
# e.g., 7B model 28GB -> 5.5GB
base_model = load_model_in_4bit("Qwen/Qwen2.5-0.5B")

# --- STEP 1: Apply LoRA Adaptation ---
# Freeze all base parameters and attach small trainable rank-matrices (LoRA)
# Only ~1% of parameters will be updated.
lora_config = LoraConfig(r=16, alpha=32, target_modules=["q_proj", "v_proj"])
model = get_peft_model(base_model, lora_config)

# --- STEP 2: Supervised Fine-Tuning (SFT) ---
# Train the LoRA adapters using QA dataset
# Data Example: {"instruction": "What is 1+1?", "output": "It is 2."}
for batch in sft_dataset:
    loss = calculate_sft_loss(model, batch) # Masking prompt, learn response
    loss.backward()
    optimizer.step()

# --- STEP 3: Preference Alignment (DPO) ---
# Refine the same LoRA adapters or train a new layer for human preference
# Data Example: {prompt, chosen, rejected}
ref_model = copy_and_freeze(model) # Reference is the SFT-ed model
for batch in dpo_dataset:
    loss = calculate_dpo_loss(model, ref_model, batch)
    loss.backward()
    optimizer.step()

# --- STEP 4: Merge and Save ---
# Combine the learned LoRA adapters back into the base model
final_model = model.merge_and_unload()
final_model.save_pretrained("./my_expert_qwen")