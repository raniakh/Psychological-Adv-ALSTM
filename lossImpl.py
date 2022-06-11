import torch


def hingeloss(output, target):
    tmp = torch.mul(output, target)
    ones = torch.ones((target.size(0), target.size(1)))
    hinge_losses = ones - tmp
    hinge_losses[hinge_losses < 0] = 0
    loss = torch.sum(hinge_losses) / output.size(0)
    return loss
