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
        _embedded, _masks, _embedded_bg = [], [], []
        voxels_with_labels = [(msk > 0).any(dim=1) for msk in masks]
        for _s, _nonempty_voxels in enumerate(voxels_with_labels):
            features = embedded[_s].reshape(-1, embedded[_s].size(-1))
            _embedded.append(features[_nonempty_voxels])
            _masks.append(masks[_s].view(-1, masks[_s].size(-1))[_nonempty_voxels])
            if self.push_away_background:
                _embedded_bg.append(features[~_nonempty_voxels])
        centroids = self._centroids(_embedded, _masks, size)
        L_v = self._variance(_embedded, _masks, centroids, size)
        L_d = self._distance(centroids, size)
        L_r = self._regularization(centroids, size)
        loss = self.alpha * L_v + self.beta * L_d + self.gamma * L_r
        if self.push_away_background:
            loss += self.gamma * self._push_zero_class(_embedded, _embedded_bg)
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


def slice_embedded(embedded, object_shape):
    embedded_lst = []
    start_slice = 0
    for obj_shape in object_shape:
        finish_slice = start_slice + obj_shape
        embedded_lst.append(embedded[start_slice:finish_slice])
        start_slice += obj_shape
    return embedded_lst


def stack_instance_dicts(dicts_lst):
    # masks_values = torch.cat([dct['masks'] for dct in dicts_lst])
    masks_values = [dct['masks'] for dct in dicts_lst]
    # size_values = torch.cat([dct['size'].repeat(dct['masks'].shape[0]) for dct in dicts_lst])
    size_values = torch.stack([dct['size'] for dct in dicts_lst])
    object_shape = torch.stack([torch.tensor(dct['object_shape']) for dct in dicts_lst])
    return masks_values, size_values, object_shape


class InstanceSegmentation(Residual3DUnet):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.loss = DiscriminativeLoss(self.hparams.delta_d, self.hparams.delta_v)

    def forward(self, batch):
        features, instseg_dct = batch
        masks, size, object_shape = stack_instance_dicts(instseg_dct)

        embedded = self.model(features).F
        # print(embedded)
        embedded = slice_embedded(embedded, object_shape)
        loss = self.loss(embedded, masks, size)
        return loss, embedded, masks

    def training_step(self, batch, batch_idx):
        loss, embedded, masks = self.forward(batch)
        self.log('train/loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, embedded, masks = self.forward(batch)
        self.log('val/loss', loss, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        embeddings = self.model(batch['input'])
        _predictions = batch.copy()
        _predictions['scene_id'] = batch_idx
        _embedding = embeddings[0]
        _semantic = batch[f'semantic_{self.hparams.labels_mapping}'][0]
        _full_size_pred = torch.zeros_like(_semantic)
        _instance = batch[self.hparams.instance_mask][0]

        _instance[_semantic == 0] = 0
        semantic_nnz_idx = _semantic > 0
        truth_inst_masks = _instance[_instance > 0]
        nnz_feat = _embedding[:, semantic_nnz_idx].T
        semantic_prediction = _semantic[semantic_nnz_idx]
        nnz_feat, semantic_prediction, truth_inst_masks = [
            _tensor.cpu().numpy() for _tensor in
            (nnz_feat, semantic_prediction, truth_inst_masks)]
        pred_instances = self.clustering_method.fit_predict(nnz_feat)
        _full_size_pred[semantic_nnz_idx] = torch.tensor(pred_instances, device=_semantic.device) + 1
        _predictions['prediction'] = _full_size_pred
        self.metrics.update_rates(pred_instances, semantic_prediction, truth_inst_masks)
        if self.predictions is not None:
            self.predictions.append(to_cpu(_predictions))
        else:
            del _predictions

    def test_epoch_end(self, outputs):
        self.metrics.compute_metrics(metrics_file=self.hparams.metrics_file)
        self.metrics.reset()
        if self.hparams.predictions_file:
            file_name = self.hparams.predictions_file
            with open(file_name, 'wb+') as f:
                pickle.dump(self.predictions, f)
        return {}
