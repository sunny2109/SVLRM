import torch
import torch.nn as nn

class Loss(nn.Module):
    """L1 loss with Charbonnier penalty function"""
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
    
    def forward(self, x, y):
        diff = x - y
        return torch.sum(torch.sqrt(diff * diff + self.eps * self.eps))