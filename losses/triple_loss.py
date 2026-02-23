import torch
import torch.nn.functional as F

def compute_triplet_loss_for_hardest_case(embeddings, labels, margin=1.0):
    """
    translated,translated ,translated torch.nonzero() translated

    Args:
        embeddings: (batch_size, embedding_dim) - translated
        labels: (batch_size,) - translated (0 translated 1)
        margin: float - translated

    Returns:
        triplet_loss: scalar - translated
    """
    device = embeddings.device
    batch_size = embeddings.size(0)

    # translated (L2translated)
    embeddings = F.normalize(embeddings, p=2, dim=1)

    # translated
    # dist_matrix[i, j] = ||embeddings[i] - embeddings[j]||^2
    # translated (a-b)^2 = a^2 + b^2 - 2ab translated
    dot_product = torch.matmul(embeddings, embeddings.t())  # (batch, batch)
    squared_norm = torch.diag(dot_product).unsqueeze(0)  # (1, batch)
    dist_matrix = squared_norm - 2.0 * dot_product + squared_norm.t()  # (batch, batch)
    dist_matrix = torch.clamp(dist_matrix, min=0.0)  # translated

    # translated (translated nonzero)
    # labels: (batch_size,) -> (batch_size, 1) translated (1, batch_size)
    labels = labels.float().view(-1, 1)  # (batch, 1)
    labels_t = labels.t()  # (1, batch)

    # positive_mask[i, j] = 1 translated i translated j translated
    positive_mask = (labels == labels_t).float()  # (batch, batch)

    # negative_mask[i, j] = 1 translated i translated j translated
    negative_mask = (labels != labels_t).float()  # (batch, batch)

    # translated (translated)
    identity = torch.eye(batch_size, device=device)
    positive_mask = positive_mask * (1 - identity)  # translated

    # translated
    num_positive_per_anchor = positive_mask.sum(dim=1, keepdim=True)  # (batch, 1)
    num_negative_per_anchor = negative_mask.sum(dim=1, keepdim=True)  # (batch, 1)

    # translated
    valid_anchors = (num_positive_per_anchor > 0) & (num_negative_per_anchor > 0)  # (batch, 1)
    valid_anchors = valid_anchors.squeeze()  # (batch,)

    if valid_anchors.sum() == 0:
        # translated,translated
        return torch.tensor(0.0, device=device, requires_grad=True)

    # ===== translated (Hard Negative Mining) =====
    # translated anchor,translated(translated)translated(translated)

    # 1. translated (translated)
    # translated,translated max translated
    masked_positive_dist = dist_matrix * positive_mask + (1 - positive_mask) * (-1e9)
    hardest_positive_dist = masked_positive_dist.max(dim=1, keepdim=True)[0]  # (batch, 1)

    # 2. translated (translated)
    # translated,translated min translated
    masked_negative_dist = dist_matrix * negative_mask + (1 - negative_mask) * 1e9
    hardest_negative_dist = masked_negative_dist.min(dim=1, keepdim=True)[0]  # (batch, 1)

    # 3. translated: max(d(a,p) - d(a,n) + margin, 0)
    triplet_loss = F.relu(hardest_positive_dist - hardest_negative_dist + margin)  # (batch, 1)

    # 4. translated anchor translated
    triplet_loss = triplet_loss.squeeze()  # (batch,)
    triplet_loss = triplet_loss * valid_anchors.float()  # translated

    # translated
    num_valid = valid_anchors.sum().float()
    if num_valid > 0:
        triplet_loss = triplet_loss.sum() / num_valid
    else:
        triplet_loss = torch.tensor(0.0, device=device, requires_grad=True)

    return triplet_loss

def compute_triplet_loss_for_all(embeddings, labels, margin=1.0):
    device = embeddings.device
    embeddings = F.normalize(embeddings, p=2, dim=1)
    batch_size = embeddings.size(0)

    # pairwise dist
    dot = embeddings @ embeddings.t()
    sq = dot.diag().unsqueeze(1)  # (B, 1)
    dist = sq - 2 * dot + sq.t()  # (B, B)
    dist = torch.clamp(dist, min=0.0)

    labels = labels.view(-1, 1)
    same = (labels == labels.t())
    diff = ~same

    # translated
    eye = torch.eye(batch_size, dtype=torch.bool, device=device)
    same = same & ~eye

    # (B, B, B): d_ap(i, j) - d_an(i, k)
    d_ap = dist.unsqueeze(2)          # (B, B, 1)
    d_an = dist.unsqueeze(1)          # (B, 1, B)

    mask_pos = same.unsqueeze(2)      # (B, B, 1)
    mask_neg = diff.unsqueeze(1)      # (B, 1, B)

    mask = mask_pos & mask_neg        # translated triplet translated

    triplet = d_ap - d_an + margin
    triplet = F.relu(triplet)

    triplet = triplet * mask            # translated 0
    num_triplet = mask.sum()

    if num_triplet == 0:
        return torch.zeros((), device=device, requires_grad=True)

    loss = triplet.sum() / num_triplet
    return loss

# ===== translated (translated,translated) =====
def compute_triplet_loss_old(embeddings, labels, margin=1):
    """
    ⚠️ translated: translated torch.nonzero() translated,translated """
    # embeddings: (batch_size, embedding_dim)
    # labels: (batch_size,)
    device = embeddings.device
    batch_size = embeddings.size(0)

    # translated
    embeddings = F.normalize(embeddings, p=2, dim=1)

    # translated
    triplet_loss = torch.tensor(0.0, device=device)

    # translated,translated
    for i in range(batch_size):
        anchor = embeddings[i]
        anchor_label = labels[i]

        # translated
        positive_indices = torch.nonzero(labels == anchor_label, as_tuple=False).view(-1)
        positive_indices = positive_indices[positive_indices != i]  # translated
        if positive_indices.numel() == 0:
            continue  # translated,translated

        positive = embeddings[positive_indices]

        # translated
        negative_indices = torch.nonzero(labels != anchor_label, as_tuple=False).view(-1)
        if negative_indices.numel() == 0:
            continue  # translated,translated

        negative = embeddings[negative_indices]

        # translated pos_dist = (anchor - positive).pow(2).sum(1)  # (num_positive,)
        neg_dist = (anchor - negative).pow(2).sum(1)  # (num_negative,)

        # translated
        losses = F.relu(pos_dist.unsqueeze(1) - neg_dist.unsqueeze(0) + margin)
        triplet_loss += losses.mean()

    # translated
    triplet_loss = triplet_loss / batch_size

    return triplet_loss
