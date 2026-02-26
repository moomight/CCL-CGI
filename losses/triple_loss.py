import torch
import torch.nn.functional as F

def compute_triplet_loss_for_hardest_case(embeddings, labels, margin=1.0):
    """

    Args:
        embeddings: (batch_size, embedding_dim) 
        labels: (batch_size,) 
        margin: float 

    Returns:
        triplet_loss: scalar 
    """
    device = embeddings.device
    batch_size = embeddings.size(0)

    embeddings = F.normalize(embeddings, p=2, dim=1)

    # dist_matrix[i, j] = ||embeddings[i] - embeddings[j]||^2
    dot_product = torch.matmul(embeddings, embeddings.t())  # (batch, batch)
    squared_norm = torch.diag(dot_product).unsqueeze(0)  # (1, batch)
    dist_matrix = squared_norm - 2.0 * dot_product + squared_norm.t()  # (batch, batch)
    dist_matrix = torch.clamp(dist_matrix, min=0.0)

    labels = labels.float().view(-1, 1)  # (batch, 1)
    labels_t = labels.t()  # (1, batch)

    positive_mask = (labels == labels_t).float()  # (batch, batch)

    negative_mask = (labels != labels_t).float()  # (batch, batch)

    identity = torch.eye(batch_size, device=device)
    positive_mask = positive_mask * (1 - identity)

    num_positive_per_anchor = positive_mask.sum(dim=1, keepdim=True)  # (batch, 1)
    num_negative_per_anchor = negative_mask.sum(dim=1, keepdim=True)  # (batch, 1)

    valid_anchors = (num_positive_per_anchor > 0) & (num_negative_per_anchor > 0)  # (batch, 1)
    valid_anchors = valid_anchors.squeeze()  # (batch,)

    if valid_anchors.sum() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)


    masked_positive_dist = dist_matrix * positive_mask + (1 - positive_mask) * (-1e9)
    hardest_positive_dist = masked_positive_dist.max(dim=1, keepdim=True)[0]  # (batch, 1)

    masked_negative_dist = dist_matrix * negative_mask + (1 - negative_mask) * 1e9
    hardest_negative_dist = masked_negative_dist.min(dim=1, keepdim=True)[0]  # (batch, 1)

    triplet_loss = F.relu(hardest_positive_dist - hardest_negative_dist + margin)  # (batch, 1)

    triplet_loss = triplet_loss.squeeze()  # (batch,)
    triplet_loss = triplet_loss * valid_anchors.float()

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

    eye = torch.eye(batch_size, dtype=torch.bool, device=device)
    same = same & ~eye

    # (B, B, B): d_ap(i, j) - d_an(i, k)
    d_ap = dist.unsqueeze(2)          # (B, B, 1)
    d_an = dist.unsqueeze(1)          # (B, 1, B)

    mask_pos = same.unsqueeze(2)      # (B, B, 1)
    mask_neg = diff.unsqueeze(1)      # (B, 1, B)

    mask = mask_pos & mask_neg

    triplet = d_ap - d_an + margin
    triplet = F.relu(triplet)

    triplet = triplet * mask
    num_triplet = mask.sum()

    if num_triplet == 0:
        return torch.zeros((), device=device, requires_grad=True)

    loss = triplet.sum() / num_triplet
    return loss

def compute_triplet_loss_old(embeddings, labels, margin=1):
    
    # embeddings: (batch_size, embedding_dim)
    # labels: (batch_size,)
    device = embeddings.device
    batch_size = embeddings.size(0)

    embeddings = F.normalize(embeddings, p=2, dim=1)

    triplet_loss = torch.tensor(0.0, device=device)

    for i in range(batch_size):
        anchor = embeddings[i]
        anchor_label = labels[i]

        positive_indices = torch.nonzero(labels == anchor_label, as_tuple=False).view(-1)
        positive_indices = positive_indices[positive_indices != i]
        if positive_indices.numel() == 0:
            continue

        positive = embeddings[positive_indices]

        negative_indices = torch.nonzero(labels != anchor_label, as_tuple=False).view(-1)
        if negative_indices.numel() == 0:
            continue

        negative = embeddings[negative_indices]

        neg_dist = (anchor - negative).pow(2).sum(1)  # (num_negative,)

        losses = F.relu(pos_dist.unsqueeze(1) - neg_dist.unsqueeze(0) + margin)
        triplet_loss += losses.mean()

    triplet_loss = triplet_loss / batch_size

    return triplet_loss
