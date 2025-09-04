import torch


def mpjpe(pred, target):
    return torch.mean(torch.norm(pred - target, dim=-1))
