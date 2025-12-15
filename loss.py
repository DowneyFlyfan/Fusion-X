import torch
import torch.nn as nn
import torch.nn.functional as F


class multi_batch_mse(nn.Module):
    """a double qsrt Loss for zero-shot pansharpening
    gt: (L, c, p, p)
    pred: (L, c, p, p)
    """

    def __init__(self):
        super().__init__()

    def forward(self, gt, pred):
        loss = (pred - gt).pow(2).mean(dim=(-1, -2)).sqrt().sum(dim=0).sqrt().sum()

        return loss


class SAMLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, gt, pred):
        t = (torch.sum(gt * gt, 1) * torch.sum(pred * pred, 1)) ** 0.5
        num = torch.sum(torch.gt(t, 0))
        angle = torch.acos(
            torch.clamp(torch.sum(gt * pred, 1) / t, min=-1, max=0.99999997)
        )
        return angle.sum() / num


class multi_batch_sam(nn.Module):
    """a patch-wise SAM Loss for zero-shot pansharpening
    gt: (bhw, c, p, p)
    pred: (bhw, c, p, p)
    """

    def __init__(self):
        super().__init__()
        self.eps = 1e-8

    def forward(self, gt, pred):
        angle = torch.acos(
            torch.clamp(
                torch.sum(gt * pred, 1)
                / (torch.sum(gt * gt, 1) * torch.sum(pred * pred, 1)) ** 0.5,
                min=-1,
                max=0.99999997,  # 是1的话角度为0.还是会出现nan
            ).mean(dim=(-1, -2))
        )

        return angle.sum()


class RegLoss(nn.Module):
    def __init__(self, lambda_reg):
        super(RegLoss, self).__init__()
        self.lambda_reg = lambda_reg

    def forward(self, residual):
        residual_flatten = residual.reshape(-1)
        residual_zoom = residual_flatten * self.lambda_reg
        loss = torch.norm(residual_zoom, p=2, dim=0)
        return loss
