from argparse import Namespace
import pickle

import torch
from torch.utils.data import DataLoader
from sklearn.cluster import MeanShift
from datasets.scannet import generate_test_dataset
from unet3d.lightning_model import Residual3DUnet
from unet3d.losses import DiscriminativeLoss
from unet3d.metrics import EvaluateInstanceSegmentationPR
from datasets.utils import stack_instance_dicts, slice_embedded
from utils.prediction import to_cpu
import MinkowskiEngine as ME
from MinkowskiEngine.utils import sparse_collate


class InstanceSegmentationResidual3DUnet(Residual3DUnet):
    def __init__(self, hparams, train_phase=True):
        super().__init__(hparams)
        if train_phase:
            self.generate_train_and_val_datasets()
            self.hparams.num_classes = self.train_dataset.num_classes
            
        push_away_background = self.hparams.__dict__.get('push_away_background', False)
        self.loss = DiscriminativeLoss(self.hparams.delta_d, self.hparams.delta_v,
                                       push_away_background=push_away_background)

    def forward(self, batch):
        features, instseg_dct = batch
        masks, size, object_shape = stack_instance_dicts(instseg_dct)

        embedded = self.model(features).F
        # print(embedded)
        embedded = slice_embedded(embedded, object_shape)

        # print('\n###############')
        # print('forward features shape:', features.shape)
        # print('embedded len:', len(embedded))
        # print('embedded[0] shape:', embedded[0].shape)
        # print('masks len:', len(masks))
        # print('masks[0] shape:', masks[0].shape)
        # print('size[:10]:', size[:10])
        # print('###############\n')

        # embedded = embedded.permute(0, 2, 3, 4, 1)

        loss = self.loss(embedded, masks, size)
        return loss, embedded, masks
        # return None, None, None

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




class InstanceSegmentationResidual3DUnetForTesting(InstanceSegmentationResidual3DUnet):
    def __init__(self, hparams):
        if isinstance(hparams, dict):
            hparams = Namespace(**hparams)
        super().__init__(hparams, train_phase=False)
        self.test_dataset = None
        self.predictions = None
        self.metrics = None
        self.clustering_method = None

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

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1,
                          shuffle=False, num_workers=self.hparams.num_workers)

    def generate_test_dataset(self, **kwargs):
        if self.hparams.predictions_file:
            self.predictions = []
        self.test_dataset = generate_test_dataset(**self.hparams.__dict__)
        self.hparams.num_classes = self.test_dataset.num_classes[0]
        self.metrics = EvaluateInstanceSegmentationPR(self.hparams)
        self.clustering_method = MeanShift(bandwidth=self.hparams.bandwidth,
                                           n_jobs=self.hparams.num_workers or None)
