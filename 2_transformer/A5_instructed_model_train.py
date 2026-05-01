# Model finetuning process for Instructed Models (e.g., LLaMA, Falcon) using a custom QA dataset.
# 
# Step 1: Train base model for learning the general knowledge and response generation (SFT).
# Step 2: Train the instructed model using dataset like "Q: What is the capital of France? A: Paris." to teach it to follow instructions and generate accurate answers.
# Step 3: Optionally, apply DPO fine-tuning to further align the model with human preferences
# Step 4: Evaluate on held-out QA pairs to measure improvements in accuracy and response quality.
#
# Example QA Pair:
# prompt   = "Q: What is the capital of France?"
# response = "A: The capital is Paris."

def calculate_sft_loss_qa(model, input_ids, labels, vocab_size):    
    # input_ids = ["Q:", "What", "is...", "A:", "The", "capital", "is", "Paris", "EOS"]
    # labels    = [-100, -100, -100, ..., "The", "capital", "is", "Paris", "EOS"] 
    # (Tokens for "Q: What is..." are masked with -100)

    # 1. Forward Pass to get logits
    # logits: [Batch=1, Seq=12, Vocab=32000]
    logits = model(input_ids).logits 

    # 2. Shift for Next Token Prediction
    # Model predicts token[i+1] given token[i]
    shift_logits = logits[:, :-1, :] 
    shift_labels = labels[:, 1:]      

    # 3. Flatten for element-wise loss
    flat_logits = shift_logits.reshape(-1, vocab_size) # [11, 32000]
    flat_labels = shift_labels.reshape(-1)            # [11]

    # 4. Filter Mask (Ignore Question tokens)
    # if flat_labels = [-100, -100, 502, 310, 120] 
    # mask = [False, False, True, True, True]
    mask = (flat_labels != -100)
    valid_logits = flat_logits[mask] # Only logits for "The", "capital", ...
    valid_labels = flat_labels[mask] # Only labels [502, 310, 120, ...]

    # 5. Manual Cross Entropy
    # A. Get log probabilities
    log_probs = log_softmax(valid_logits, dim=-1)

    # B. Gather log-prob of the ground-truth answer tokens
    # e.g., picked_log_prob = log_probs[0, 502] (log-prob of "The")
    picked_log_probs = gather(log_probs, dim=-1, index=valid_labels.unsqueeze(-1))

    # C. Negative Log Likelihood
    # loss = -log(0.8) -> 0.22
    token_losses = -1.0 * picked_log_probs

    return sum(token_losses) / len(token_losses)