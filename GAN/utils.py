import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import init


def initialize_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose2d):
        init.xavier_uniform_(m.weight.data)


def bce_loss(logits, target):
    neg_abs = - logits.abs()
    loss = logits.clamp(min=0) - logits * target + (1 + neg_abs.exp()).log()
    return loss.mean()


def discriminator_loss(logits_real, logits_fake):
    return bce_loss(logits_real, torch.ones_like(logits_real)) + \
           bce_loss(logits_fake, torch.zeros_like(logits_fake))


def generator_loss(logits_fake):
    return bce_loss(logits_fake, torch.ones_like(logits_fake))


def sample_noise(batch_size, dim):
    return torch.rand(batch_size, dim) * 2.0 - 1.0

def get_optimizer(model):
    return optim.Adam(model.parameters(), 1e-3, (0.5, 0.999))

class Flatten(nn.Module):
    def forward(self, x):
        N, C, H, W = x.size()
        return x.view(N, -1)


class Unflatten(nn.Module):
    def __init__(self, N, C, H, W):
        super(Unflatten, self).__init__()
        self.N = N
        self.C = C
        self.H = H
        self.W = W

    def forward(self, x):
        return x.view(self.N, self.C, self.H, self.W)
