import torch
import torch.nn.functional as F


def compute_triplet_loss(embeddings, labels, margin=1):
    # embeddings: (batch_size, embedding_dim)
    # labels: (batch_size,)
    device = 'cuda'
    batch_size = embeddings.size(0)

    # 归一化嵌入向量
    embeddings = F.normalize(embeddings, p=2, dim=1)

    # 初始化损失
    triplet_loss = torch.tensor(0.0, device=device)

    # 遍历每个样本，构造三元组
    for i in range(batch_size):
        anchor = embeddings[i]
        anchor_label = labels[i]

        # 从同一类别中选择正样本
        positive_indices = torch.nonzero(labels == anchor_label).squeeze()
        positive_indices = positive_indices[positive_indices != i]  # 排除自身
        if len(positive_indices) == 0:
            continue  # 如果没有正样本，跳过

        positive = embeddings[positive_indices]

        # 从不同类别中选择负样本
        negative_indices = torch.nonzero(labels != anchor_label).squeeze()
        if len(negative_indices) == 0:
            continue  # 如果没有负样本，跳过

        negative = embeddings[negative_indices]

        # 计算锚点与正样本、负样本之间的距离
        pos_dist = (anchor - positive).pow(2).sum(1)  # (num_positive,)
        neg_dist = (anchor - negative).pow(2).sum(1)  # (num_negative,)

        # 计算损失
        losses = F.relu(pos_dist.unsqueeze(1) - neg_dist.unsqueeze(0) + margin)
        triplet_loss += losses.mean()

    # 取平均损失
    triplet_loss = triplet_loss / batch_size

    return triplet_loss
