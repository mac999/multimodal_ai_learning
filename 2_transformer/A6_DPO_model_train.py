# DPO Loss Calculation for QA Tasks
# policy_model: The model being trained (the "student")
# ref_model: A frozen reference model (the "teacher" or SFT baseline)
# 
# Example Preference Pair:
# prompt   = "Q: Who wrote 'Romeo and Juliet'?"
# chosen   = "A: William Shakespeare wrote it." (Correct)
# rejected = "A: J.K. Rowling wrote it." (Wrong/Hallucination)

def calculate_dpo_loss_qa(policy_model, ref_model, batch, beta=0.1):
    # 1. Policy Model Log-Probs (Learning model)
    # policy_lp_w: log P("Shakespeare..." | "Who wrote...") = -0.3 (High)
    # policy_lp_l: log P("Rowling..." | "Who wrote...")     = -5.8 (Low)
    policy_lp_w = get_sequence_logprob(policy_model, batch['prompt'], batch['chosen'])
    policy_lp_l = get_sequence_logprob(policy_model, batch['prompt'], batch['rejected'])

    # 2. Reference Model Log-Probs (Frozen SFT baseline)
    # ref_lp_w: log P_ref("Shakespeare...") = -1.5
    # ref_lp_l: log P_ref("Rowling...")     = -2.1
    # (The SFT model is unsure, probabilities are similar)
    with no_grad():
        ref_lp_w = get_sequence_logprob(ref_model, batch['prompt'], batch['chosen'])
        ref_lp_l = get_sequence_logprob(ref_model, batch['prompt'], batch['rejected'])

    # 3. Relative Log Ratios (Current vs Baseline)
    # chosen_ratio   = (-0.3) - (-1.5) = +1.2 (Policy is much more confident in correct answer)
    # rejected_ratio = (-5.8) - (-2.1) = -3.7 (Policy is much more doubtful of wrong answer)
    chosen_log_ratio = policy_lp_w - ref_lp_w
    rejected_log_ratio = policy_lp_l - ref_lp_l

    # 4. Margin Calculation
    # margin = 1.2 - (-3.7) = 4.9
    # scaled_logits = 0.1 * 4.9 = 0.49
    margin = chosen_log_ratio - rejected_log_ratio
    scaled_logits = beta * margin
    
    # 5. DPO Loss: -log(sigmoid(margin))
    # Higher margin leads to lower loss
    dpo_loss = -1.0 * log_sigmoid(scaled_logits).mean()

    return dpo_loss