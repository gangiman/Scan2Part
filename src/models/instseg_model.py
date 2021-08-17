import pickle

import torch
from torch import nn as nn
from src.models.lightning_model import Residual3DUnet


class DiscriminativeLoss(nn.Module):
    def __init__(self, delta_d, delta_v, delta_p=1,
                 alpha=1.0, beta=1.0, gamma=0.001):
        super(DiscriminativeLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta_d = delta_d
        self.delta_v = delta_v
        self.delta_p = delta_p

    def forward(self, embedded, masks, size):
        variance_loss = 0.0
        distance_loss = 0.0
        reg_loss = 0.0
        batch_size = len(embedded)
        for n, _embed, _mask in zip(size, embedded, masks):
            num_points = _embed.size(0)
            inst_mask = _mask.unsqueeze(2)
            masked_embeddings = _embed.unsqueeze(1).expand(-1, n, -1) * inst_mask
            mu = masked_embeddings.sum(0) / inst_mask.sum(0)
            centroids = mu
            # Convert input into the same size
            mu = centroids.unsqueeze(0).expand(num_points, -1, -1)
            x = _embed.unsqueeze(1).expand(-1, n, -1)
            # Calculate intra pull force
            var = torch.norm(x - mu, 2, dim=2)
            var = torch.clamp(var - self.delta_v, min=0.0) ** 2
            var = var * inst_mask[:, :, 0]
            variance_loss += var.sum() / inst_mask.sum()
            # calculating distance loss
            if n > 1:
                # continue
                mu = centroids
                mu_a = mu.unsqueeze(1).expand(-1, n, -1)
                mu_b = mu_a.permute(1, 0, 2)
                diff = mu_a - mu_b
                norm = torch.norm(diff, 2, dim=2)
                margin = 2 * self.delta_d * (1.0 - torch.eye(n))
                margin = margin.to(centroids.device)
                distance = torch.sum(torch.clamp(margin - norm, min=0.0) ** 2)  # hinge loss
                distance /= float(n * (n - 1))
                distance_loss += distance
            # calculating regularisation
            norm = torch.norm(centroids, 2, dim=1)
            reg_loss += norm.mean()
        return (self.alpha * variance_loss + self.beta * distance_loss + self.gamma * reg_loss) / batch_size


class InstanceSegmentation(Residual3DUnet):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.loss = DiscriminativeLoss(self.hparams.delta_d, self.hparams.delta_v)

    def shared_step(self, batch):
        embedded, dict_of_lists = self.forward(batch)
        loss = self.loss(embedded, dict_of_lists['object'], dict_of_lists['object_size'])
        return loss, embedded, dict_of_lists

    def training_step(self, batch, batch_idx):
        loss, *_ = self.shared_step(batch)
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, embedded, masks_dict = self.shared_step(batch)
        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return {'embedded': embedded, **masks_dict}

    def test_step(self, batch, batch_idx):
        embedded, masks_dict = self.forward(batch)
        return {'embedded': embedded, **masks_dict}
