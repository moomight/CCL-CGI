from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch
import torch.nn.functional as F


def contrastive_loss(temp, embedding, label):
    """
    Optimized contrastive loss calculation with GPU acceleration
    """
    # Normalize embeddings
    embedding = F.normalize(embedding, p=2, dim=1)

    # Cosine similarity between embeddings (matrix multiplication)
    cosine_sim = torch.mm(embedding, embedding.t())

    # Remove diagonal elements (in-place operation for better efficiency)
    cosine_sim.fill_diagonal_(0)

    # Apply temperature and exponentiate
    cosine_sim = torch.exp(cosine_sim / temp)

    # Row-wise sum of the cosine similarity matrix
    row_sum = torch.sum(cosine_sim, dim=1, keepdim=True)

    # Create label mask to identify positive pairs (i.e., pairs with the same label)
    label_mask = (label.unsqueeze(1) == label.unsqueeze(0)).float().to('cuda')
    label_mask.fill_diagonal_(0)  # Remove self-comparisons

    # Calculate the log of the normalized similarities for positive pairs
    log_sim = torch.log(cosine_sim / row_sum)

    # Apply the label mask to select positive pairs and calculate the contrastive loss
    positive_pairs = log_sim * label_mask
    n_i = torch.sum(label_mask, dim=1)  # Number of positive pairs for each sample

    # Handle division by zero for samples with no positive pairs
    loss = torch.where(n_i > 0, -torch.sum(positive_pairs, dim=1) / n_i, torch.tensor(0.0, device='cuda'))

    # Return mean contrastive loss
    return loss.mean()


def compute_contrastive_loss(temp, embeddings, labels):
    # embeddings: (batch_size, embedding_dim)
    # labels: (batch_size, 1)
    batch_size = embeddings.size(0)

    # 步骤 1：L2 归一化嵌入向量
    embeddings = F.normalize(embeddings, p=2, dim=1)

    # 步骤 2：计算相似度矩阵
    similarity_matrix = torch.matmul(embeddings, embeddings.T)  # (batch_size, batch_size)

    # 步骤 3：处理对角线元素
    mask = torch.eye(batch_size, dtype=torch.bool).to('cuda')
    similarity_matrix = similarity_matrix.masked_fill(mask, -9e15)

    temperature = temp
    # 步骤 4：计算 logits
    logits = similarity_matrix / temperature

    # 步骤 5：使用 logsumexp 提高数值稳定性
    max_logits, _ = torch.max(logits, dim=1, keepdim=True)
    log_sum_exp = max_logits + torch.log(torch.exp(logits - max_logits).sum(dim=1, keepdim=True))

    # 步骤 6：计算 log_prob
    log_prob = logits - log_sum_exp  # (batch_size, batch_size)

    # 步骤 7：创建标签矩阵
    labels = labels.view(-1)
    labels_matrix = labels.unsqueeze(1) == labels.unsqueeze(0)  # (batch_size, batch_size)
    positive_mask = labels_matrix.float().to('cuda')
    negative_mask = 1 - positive_mask

    # 步骤 8：计算每个样本的正样本数量，避免除零
    epsilon = 1e-8
    num_positives_per_row = positive_mask.sum(dim=1)
    num_positives_per_row = torch.clamp(num_positives_per_row, min=1)

    # 步骤 9：计算平均对数概率
    mean_log_prob_pos = (positive_mask * log_prob).sum(dim=1) / num_positives_per_row

    # 步骤 10：计算对比损失
    contrastive_loss = -mean_log_prob_pos.mean()

    return contrastive_loss

