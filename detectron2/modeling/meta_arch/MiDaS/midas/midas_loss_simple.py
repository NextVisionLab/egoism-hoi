import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.filters import laplacian
from kornia.losses import ssim_loss
from kornia.enhance.normalize import normalize_min_max

class DepthLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ssim_loss_weight = 0.85
        self.l1_loss_weight = 0.9
        self.edge_loss_weight = 0.9

    def forward(self, pred, target):
        # Edges
        pred = torch.unsqueeze(pred, 1)
        target = torch.unsqueeze(target, 1)
        gt_edges = normalize_min_max(laplacian(target, kernel_size=3))
        pred_edges = normalize_min_max(laplacian(pred, kernel_size=3))

        #edge_loss = F.binary_cross_entropy(input=pred_edges, target=gt_edges)
        edge_loss = ssim_loss(pred_edges, gt_edges, window_size=3, reduction="mean")

        # Structural similarity (SSIM) index
        ssim_loss_ = ssim_loss(pred, target, window_size=3, reduction="mean", max_val=255.0)

        # Point-wise depth
        l1_loss = F.l1_loss(pred, target)

        loss = ((self.ssim_loss_weight * ssim_loss_) + (self.l1_loss_weight * l1_loss) + (self.edge_loss_weight * edge_loss))
        return loss