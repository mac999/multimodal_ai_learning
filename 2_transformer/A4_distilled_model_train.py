# Distilled Model Training Loop pseudocode
# 
import torch
import torch.nn as nn

def softmax(logits):
    exp = exp_fn(logits - max_fn(logits))   # numerical stability
    return exp / sum_fn(exp)

def cross_entropy(student_logits, labels):
    """
    Cross Entropy = correct answer matching
    labels: correct token id (others = -100 ignored)
    """
    probs = softmax(student_logits)

    loss = 0
    count = 0

    for i in range(len(labels)):
        if labels[i] != -100:  # masked positions only
            loss += -log_fn(probs[i][labels[i]])
            count += 1

    return loss / max(count, 1)


def kl_divergence(p_teacher, p_student):
    """
    KL Divergence = distribution difference
    """
    eps = 1e-12
    loss = 0

    for i in range(len(p_teacher)):
        for j in range(len(p_teacher[i])):  # over vocab
            pt = max(p_teacher[i][j], eps)
            ps = max(p_student[i][j], eps)

            loss += pt * (log_fn(pt) - log_fn(ps))

    return loss / len(p_teacher)


def cosine_similarity(h1, h2):
    """
    Cosine similarity between two vectors
    """
    dot = dot_product(h1, h2)
    norm = l2_norm(h1) * l2_norm(h2)
    return dot / max(norm, 1e-12)


# Training loop
# Teacher: pretrained, frozen
teacher.eval()

# Student: trainable
student.train()

for epoch in range(E):
    for x in dataloader:

        # 1) Mask tokens (15%)
        x_masked, labels = mask_tokens(x)

        # 2) Forward pass
        z_T, h_T = teacher(x_masked)   # teacher logits, hidden
        z_S, h_S = student(x_masked)   # student logits, hidden

        # 3) Soft probabilities with temperature
        p_T = softmax(z_T / T)
        p_S = softmax(z_S / T)

        # 4) Losses

        # (1) Cross Entropy → correct answer learning
        loss_ce = cross_entropy(z_S, labels)

        # (2) KL Divergence → mimic teacher distribution
        loss_kl = kl_divergence(p_T, p_S)

        # (3) Cosine Loss → match internal representation
        loss_cos = 1 - cosine_similarity(h_T, h_S)

        # 5) Total loss
        loss = a * loss_ce + b * loss_kl + c * loss_cos

        # 6) Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()