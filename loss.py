import torch
import torch.nn as nn
import torch.nn.functional as F


class Alignment_loss(nn.Module):
    def __init__(self, gamma, batch_size, device):
        super(Alignment_loss, self).__init__()
        self.gamma = gamma
        self.batch_size = batch_size
        self.device = device

    def forward(self, outfeature, trainset):
        h = outfeature.to(device=self.device)
        set = torch.as_tensor(trainset).to(device=self.device)

        def _cosine(x):
            dot1 = torch.matmul(x[0], x[1], axes=1)
            dot2 = torch.matmul(x[0], x[0], axes=1)
            dot3 = torch.matmul(x[1], x[1], axes=1)
            max_ = torch.maximum(torch.sqrt(dot2 * dot3), torch.tensor(torch.finfo(dot1.dtype).eps))
            return dot1 / max_

        l, r, fl, fr = [h[set[:, 0]], h[set[:, 1]], h[set[:, 2]], h[set[:, 3]]]

        l1_dist = torch.norm(l - r, p=1, dim=-1, keepdim=True)

        loss = torch.clamp(self.gamma + l1_dist - torch.norm(l - fr, p=1, dim=-1, keepdim=True), min=0) + torch.clamp(
            self.gamma + l1_dist - torch.norm(fl - r, p=1, dim=-1, keepdim=True), min=0)

        loss_avg = torch.sum(loss, dim=0, keepdim=True)

        return loss_avg
