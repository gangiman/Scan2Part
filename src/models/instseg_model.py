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
        centroids = self._centroids(embedded, masks, size)
        L_v = self._variance(embedded, masks, centroids, size)
        L_d = self._distance(centroids, size)
        L_r = self._regularization(centroids, size)
        loss = self.alpha * L_v + self.beta * L_d + self.gamma * L_r
        return loss

    def _centroids(self, embedded, masks, size):
        batch_size = len(embedded)
        embedding_size = embedded[0].size(-1)
        K = masks[0].size(-1)
        masked_embeddings = []
        for _embed, _mask in zip(embedded, masks):
            masked_embeddings.append(
                _embed.unsqueeze(1).expand(-1, K, -1) * _mask.unsqueeze(2))
        centroids = []
        for i in range(batch_size):
            n = size[i]
            mu = masked_embeddings[i][:, :n].sum(0) / masks[i].unsqueeze(2)[:, :n].sum(0)
            if K > n:
                m = int(K - n)
                filled = torch.zeros(m, embedding_size)
                filled = filled.to(embedded[0].device)
                mu = torch.cat([mu, filled], dim=0)
            centroids.append(mu)
        centroids = torch.stack(centroids)
        return centroids

    def _variance(self, embedded, masks, centroids, size):
        loss = 0.0
        batch_size = len(embedded)
        for _embed, _mask, _centroids, n in zip(embedded, masks, centroids, size):
            num_points = _embed.size(0)
            K = _mask.size(1)
            # Convert input into the same size
            mu = _centroids.unsqueeze(0).expand(num_points, -1, -1)
            x = _embed.unsqueeze(1).expand(-1, K, -1)
            # Calculate intra pull force
            var = torch.norm(x - mu, 2, dim=2)
            var = torch.clamp(var - self.delta_v, min=0.0) ** 2
            var = var * _mask
            loss += torch.sum(var[:, :n]) / torch.sum(_mask[:, :n])
        loss /= batch_size
        return loss

    def _distance(self, centroids, size):
        batch_size = centroids.size(0)
        loss = 0.0
        for i in range(batch_size):
            n = size[i]
            if n <= 1:
                continue
            mu = centroids[i, :n, :]
            mu_a = mu.unsqueeze(1).expand(-1, n, -1)
            mu_b = mu_a.permute(1, 0, 2)
            diff = mu_a - mu_b
            norm = torch.norm(diff, 2, dim=2)
            margin = 2 * self.delta_d * (1.0 - torch.eye(n))
            margin = margin.to(centroids.device)
            distance = torch.sum(torch.clamp(margin - norm, min=0.0) ** 2)  # hinge loss
            distance /= float(n * (n - 1))
            loss += distance
        loss /= batch_size
        return loss

    def _regularization(self, centroids, size):
        batch_size = centroids.size(0)
        loss = 0.0
        for i in range(batch_size):
            n = size[i]
            mu = centroids[i, :n, :]
            norm = torch.norm(mu, 2, dim=1)
            loss += torch.mean(norm)
        loss /= batch_size
        return loss


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
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, embedded, masks_dict = self.shared_step(batch)
        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {'embedded': embedded, **masks_dict}

    def test_step(self, batch, batch_idx):
        embedded, masks_dict = self.forward(batch)
        return {'embedded': embedded, **masks_dict}
