import torch
import torch.nn as nn
import torch.nn.functional as F


# SSIM Loss
# cite: https://zenn.dev/taikiinoue45/articles/bf7d2314ab4d10
class SSIMLoss(nn.Module):

    def __init__(self, channels: int = 3, kernel_size: int = 11, sigma: float = 1.5) -> None:
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.gaussian_kernel = self._create_gaussian_kernel(self.kernel_size, self.sigma)

    def forward(self, x: torch.Tensor, y: torch.Tensor, as_loss: bool = True) -> torch.Tensor:
        if not self.gaussian_kernel.is_cuda:
            self.gaussian_kernel = self.gaussian_kernel.to(x.device)
        ssim_map = self._ssim(x, y)
        if as_loss:
            return 1 - ssim_map.mean()
        else:
            return ssim_map

    def _ssim(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        ux = F.conv2d(x, self.gaussian_kernel, padding=self.kernel_size // 2, groups=self.channels)
        uy = F.conv2d(y, self.gaussian_kernel, padding=self.kernel_size // 2, groups=self.channels)
        uxx = F.conv2d(x * x, self.gaussian_kernel, padding=self.kernel_size // 2, groups=self.channels)
        uyy = F.conv2d(y * y, self.gaussian_kernel, padding=self.kernel_size // 2, groups=self.channels)
        uxy = F.conv2d(x * y, self.gaussian_kernel, padding=self.kernel_size // 2, groups=self.channels)
        vx = uxx - ux * ux
        vy = uyy - uy * uy
        vxy = uxy - ux * uy
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        numerator = (2 * ux * uy + c1) * (2 * vxy + c2)
        denominator = (ux ** 2 + uy ** 2 + c1) * (vx + vy + c2)
        return numerator / (denominator + 1e-12)

    def _create_gaussian_kernel(self, kernel_size: int, sigma: float) -> torch.Tensor:
        start = (1 - kernel_size) / 2
        end = (1 + kernel_size) / 2
        kernel_1d = torch.arange(start, end, step=1, dtype=torch.float)
        kernel_1d = torch.exp(-torch.pow(kernel_1d / sigma, 2) / 2)
        kernel_1d = (kernel_1d / kernel_1d.sum()).unsqueeze(dim=0)
        kernel_2d = torch.matmul(kernel_1d.t(), kernel_1d)
        kernel_2d = kernel_2d.expand(self.channels, 1, kernel_size, kernel_size).contiguous()
        return kernel_2d


# the loss function for VAE
class VAELoss(nn.Module):

    def __init__(self, channels: int = 3, alpha: float = 0.1):
        super(VAELoss, self).__init__()
        self.channels = channels
        self.alpha = alpha

    def forward(self, x_reconstruted, x_input, mu, lnvar):
        if self.channels == 1:
            rec = F.binary_cross_entropy(x_input, x_reconstruted, reduction='mean')
        else:
            rec = torch.mean(torch.abs(x_reconstruted - x_input))
        kl = -0.5 * torch.mean(1 + lnvar - mu**2 - torch.exp(lnvar))
        return rec + self.alpha * kl


# the loss function for VAE（SSIM Loss var）
class VAESSIMLoss(nn.Module):

    def __init__(self, channels: int = 3, alpha: float = 0.1):
        super(VAESSIMLoss, self).__init__()
        self.ssim = SSIMLoss(channels=channels)
        self.alpha = alpha

    def forward(self, x_reconstruted, x_input, mu, lnvar):
        rec = self.ssim(x_reconstruted, x_input)
        kl = -0.5 * torch.mean(1 + lnvar - mu**2 - torch.exp(lnvar))
        return rec + self.alpha * kl


# the loss function for GAN
class GANLoss(nn.Module):

    def __init__(self, label_smoothing=False, use_sigmoid=True):
        super(GANLoss, self).__init__()
        if use_sigmoid:
            self.bce = nn.BCEWithLogitsLoss()
        else:
            self.bce = nn.BCELoss()
        self.a = 0.9 if label_smoothing else 1.0

    def G_loss(self, y):
        gt = torch.ones_like(y).to(y.device)
        return self.bce(y, gt)

    def D_loss(self, y, as_real=True):
        if as_real:
            gt = self.a * torch.ones_like(y).to(y.device)
        else:
            gt = torch.zeros_like(y).to(y.device)
        return self.bce(y, gt)

    def forward(self, y, as_real=True):
        return self.D_loss(y, as_real)

# the loss function for GAN（hinge loss var）
class GANHingeLoss(nn.Module):

    def __init__(self):
        super(GANHingeLoss, self).__init__()

    def G_loss(self, y):
        return -torch.mean(y)

    def D_loss(self, y, as_real=True):
        if as_real:
            return torch.mean(F.relu(1 - y))
        else:
            return torch.mean(F.relu(1 + y))

    def forward(self, y, as_real=True):
        return self.D_loss(y, as_real)


# Triplet Margin Loss based on Cosine Distance
class CosineTripletMarginLoss(nn.Module):

    def __init__(self, margin=0.3, scale=16.0):
        super(CosineTripletMarginLoss, self).__init__()
        self.margin = margin
        self.scale = scale

    def forward(self, anc, pos, *negs):
        c = -torch.sum(anc * pos, dim=1) + self.margin
        for neg in negs:
            c = c + torch.sum(anc * neg, dim=1)
        return torch.mean(torch.log(1 + torch.exp(self.scale * c)))
