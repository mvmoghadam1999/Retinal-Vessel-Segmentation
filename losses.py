import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceBCELoss(nn.Module):
    """Binary Cross-Entropy + Dice loss."""

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs = probs.reshape(-1)
        targets = targets.float().reshape(-1)

        intersection = (probs * targets).sum()
        dice_loss = 1 - (
            (2 * intersection + self.smooth)
            / (probs.sum() + targets.sum() + self.smooth)
        )
        bce = F.binary_cross_entropy(probs, targets)
        return bce + dice_loss


def iou_loss(y_true, y_pred, smooth=1.0):
    intersection = torch.sum(torch.abs(y_true * y_pred), dim=(1, 2, 3))
    union = (
        torch.sum(y_true, dim=(1, 2, 3))
        + torch.sum(y_pred, dim=(1, 2, 3))
        - intersection
    )
    return 1 - torch.mean((intersection + smooth) / (union + smooth))


def focal_loss(y_true, y_pred, gamma=2.0, alpha=4.0, epsilon=1e-9):
    y_true = y_true.float()
    y_pred = y_pred.float()
    model_out = y_pred + epsilon
    ce = y_true * -torch.log(model_out)
    weight = y_true * torch.pow(1 - model_out, gamma)
    focal = alpha * weight * ce
    return torch.mean(torch.max(focal, dim=1)[0])


def gaussian_kernel(size, sigma):
    coords = torch.arange(size).float() - size // 2
    coords = coords.unsqueeze(0).repeat(size, 1)
    gaussian = torch.exp(-(coords ** 2 + coords.t() ** 2) / (2 * sigma ** 2))
    return gaussian / gaussian.sum()


class SSIM(nn.Module):
    """Optional SSIM loss retained from the original experiments."""

    def __init__(self, window_size=11, size_average=True):
        super().__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.register_buffer(
            "window",
            gaussian_kernel(window_size, 1.5).unsqueeze(0).unsqueeze(0),
        )

    def forward(self, img1, img2):
        _, channel, _, _ = img1.size()

        if channel != self.channel or self.window.size(0) != channel:
            window = gaussian_kernel(
                self.window_size, 1.5
            ).unsqueeze(0).unsqueeze(0).repeat(channel, 1, 1, 1)
            self.window = window.to(img1.device).type_as(img1)
            self.channel = channel

        window = self.window.to(img1.device).type_as(img1)

        mu1 = F.conv2d(img1, window, padding=self.window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=self.window_size // 2, groups=channel)

        mu1_sq, mu2_sq, mu1_mu2 = mu1.pow(2), mu2.pow(2), mu1 * mu2
        sigma1_sq = F.conv2d(img1 * img1, window, padding=self.window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=self.window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=self.window_size // 2, groups=channel) - mu1_mu2

        c1, c2 = 0.01 ** 2, 0.03 ** 2
        ssim_map = (
            (2 * mu1_mu2 + c1) * (2 * sigma12 + c2)
            / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
        )

        if self.size_average:
            return 1 - ssim_map.mean()
        return 1 - ssim_map.mean(1).mean(1).mean(1)


def ssim_loss(y_true, y_pred):
    return SSIM()(y_true, y_pred)


def dice_coef(y_true, y_pred, smooth=1e-9):
    intersection = torch.sum(y_true * y_pred, dim=(2, 3))
    union = torch.sum(y_true, dim=(2, 3)) + torch.sum(y_pred, dim=(2, 3))
    return torch.mean((2 * intersection + smooth) / (union + smooth))
