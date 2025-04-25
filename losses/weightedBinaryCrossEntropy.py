import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightedBinaryCrossEntropy(nn.Module):
    def __init__(self, pos_ratio):
        super(WeightedBinaryCrossEntropy, self).__init__()
        self.pos_ratio = pos_ratio # 正类比例
        self.neg_ratio = 1.0 - pos_ratio
        self.sample_weights = torch.tensor(self.neg_ratio / self.pos_ratio, dtype=torch.float32)

    def forward(self, y_pred, y_true):
        y = y_true.float()
        epsilon = torch.finfo(y_pred.dtype).eps
        y_pred = torch.clamp(y_pred, epsilon, 1 - epsilon)
        y_pred = torch.log(y_pred / (1 - y_pred))  # 解码

        loss = F.binary_cross_entropy_with_logits(y_pred, y, weight=self.sample_weights)
        return torch.mean(loss * self.pos_ratio, dim=-1)

# 示例用法
# criterion = WeightedBinaryCrossEntropy(pos_ratio=0.2)
# loss = criterion(y_true, y_pred)


class W_BCEWithLogitsLoss(torch.nn.Module):

    def __init__(self, w_p=None, w_n=None):
        super(W_BCEWithLogitsLoss, self).__init__()

        self.w_p = w_p
        self.w_n = w_n

    def forward(self, pred, labels, epsilon=1e-7):
        # ps = torch.sigmoid(logits.squeeze())
        ps = pred
        loss_pos = -1 * torch.mean(self.w_p * labels * torch.log(ps + epsilon))
        loss_neg = -1 * torch.mean(self.w_n * (1 - labels) * torch.log((1 - ps) + epsilon))

        return loss_pos + loss_neg
