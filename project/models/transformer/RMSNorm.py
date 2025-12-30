import torch.nn as nn
import torch


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization
    通常用于大模型 (LLaMA, InternVL 等)
    """

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # 计算均方根
        var = torch.mean(x ** 2, dim=-1, keepdim=True)
        x_normed = x * torch.rsqrt(var + self.eps)
        return self.weight * x_normed
