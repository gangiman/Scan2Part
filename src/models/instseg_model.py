import pickle

import torch
from src.models.lightning_model import Residual3DUnet
from unet3d.losses import DiscriminativeLoss

# from datasets.utils import stack_instance_dicts, slice_embedded
# from utils.prediction import to_cpu


def slice_embedded(embedded, object_shape):
    embedded_lst = []
    start_slice = 0
    for obj_shape in object_shape:
        finish_slice = start_slice + obj_shape
        embedded_lst.append(embedded[start_slice:finish_slice])
        start_slice += obj_shape
    return embedded_lst


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
        print('\n##################')
        print('train loss:', loss.item())
        print('##################\n')
        return {'loss': loss, 'log': {'train_loss': loss}}

    def validation_step(self, batch, batch_idx):
        loss, embedded, masks = self.forward(batch)
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss,}
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}

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
