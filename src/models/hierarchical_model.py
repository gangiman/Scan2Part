import json
import pickle
from typing import List, Dict, Optional
from collections import defaultdict

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.utils.hierarchy import ComputeHPredictions
from src.datamodules.transforms import MapInstancesToSemanticLabels
from src.models.lightning_model import Residual3DUnet


class RecursiveHeadLoss(nn.Module):
    def __init__(
        self,
        name: str = 'ROOT',
        set_id: int = 0,
        children: List = None,
        max_lod: int = None,
        semantic_key: str = 'semantic',
        f_maps: int = 32,
        lod: int = 1,
        **kwargs
    ):
        super().__init__()
        self.node_name = name
        self.set_id = set_id
        self.sub_heads = nn.ModuleDict()
        if children is not None and children:
            num_classes = len(children)
            children = sorted(children, key=lambda x: int(x['set_id']))
            self.num_classes = num_classes
            if num_classes > 1:
                self.lod = lod
                self.semantic_key = f"{semantic_key}_{lod}"
                self.final_layer = nn.Linear(f_maps, num_classes, bias=False)
                self.criterion = nn.CrossEntropyLoss()
                children_set_ids = torch.tensor([int(_child['set_id']) for _child in children])
                label_mapping = torch.zeros(children_set_ids.max() + 1, dtype=torch.int64)
                label_mapping[children_set_ids] = torch.arange(children_set_ids.shape[0])
                self.label_mapping = label_mapping
            if lod < max_lod:
                for _child in children:
                    if _child.get('children', False) and len(_child['children']) > 1:
                        self.sub_heads[f"{lod + 1}_{_child['set_id']}"] = RecursiveHeadLoss(
                            **_child, lod=lod+1, max_lod=max_lod,
                            semantic_key=semantic_key, f_maps=f_maps)
        if len(self.sub_heads) == 0:
            self.sub_heads = None

    def forward(self,
                features: List[torch.Tensor],
                batch: Dict[str, List[torch.Tensor]],
                masks: Optional[List[torch.Tensor]] = None):
        if masks is None:
            masks = [torch.ones(_f.shape[0], dtype=torch.bool) for _f in features]
        logits = [self.final_layer(_features[_mask]) for _mask, _features in zip(masks, features)]
        target = batch[self.semantic_key]
        masked_targets = [_target[_mask] for _mask, _target in zip(masks, target)]
        mapped_label_targets = [self.label_mapping[_target] for _target in masked_targets]
        loss = torch.stack([
            self.criterion(_logits, _target)
            for _logits, _target in zip(logits, mapped_label_targets)
        ]).sum()
        full_logits = {f"{self.lod}_{self.set_id}": (logits, mapped_label_targets, masked_targets)}
        if self.sub_heads is not None:
            for _k, _sub_head in self.sub_heads.items():
                features_subset = []
                masks_subset = []
                targets_subset = defaultdict(list)
                set_id = int(_sub_head.set_id)
                for _idx, _target in enumerate(target):
                    if set_id in _target:
                        features_subset.append(features[_idx])
                        # masking by gt give to sub_level
                        masks_subset.append(masks[_idx] & (set_id == _target))
                        for _key, _targets in batch.items():
                            targets_subset[_key].append(_targets[_idx])
                if features_subset:
                    sub_head_loss, sub_logits = _sub_head(features_subset, targets_subset, masks_subset)
                    loss += sub_head_loss
                    full_logits.update(sub_logits)
        return loss, full_logits


class HierarchicalModel(Residual3DUnet):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        with open(self.hparams.hierarchy_file) as fh:
            hierarchy_0k = json.loads(fh.read())

        self.recursive_heads = RecursiveHeadLoss(
            name='ROOT', set_id=0,
            children=hierarchy_0k['ROOT'],
            f_maps=self.hparams.f_maps,
            max_lod=self.hparams.max_lod,
            semantic_key=self.hparams.semantic_key)

    def shared_step(self, batch):
        embedded, dict_of_lists = self.forward(batch)
        loss, logits = self.recursive_heads(embedded, dict_of_lists)
        return loss, logits, dict_of_lists

    def training_step(self, batch, batch_idx):
        loss, *_ = self.shared_step(batch)
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, head_logits, masks_dict = self.shared_step(batch)
        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {'head_logits': head_logits, **masks_dict}


class HierarchicalHeadModelForTesting(HierarchicalModel):
    def __init__(self, hparams):
        self.test_dataset = None
        super().__init__(hparams, train_phase=False)
        self.predictions = []
        self.test_label_mappings = {}
        self.test_confusion_matrix = {}
        self.top_down_metrics = None
        self.sm = nn.Softmax(dim=1)
        self.lsm = nn.LogSoftmax(dim=1)
        self.remaps = {}
        self.max_lod = None

    def test_step(self, batch, batch_idx):
        self.forward(batch, batch_idx, compute_metrics=True)
        return {}

    def write_metrics_file(self, metrics):
        import yaml
        metrics_file = self.hparams.metrics_file
        if metrics_file is not None:
            with open(metrics_file, 'w+') as fh:
                yaml.dump(metrics, fh)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, collate_fn=collate_heads,
                          shuffle=False, num_workers=self.hparams.num_workers)

    def forward(self, batch, batch_idx, compute_metrics=True):
        base_batch, head_batch, index_batch, targets_batch = batch
        embedded = self.model(base_batch["input"])
        head_id = 0
        _head = f"head_{head_id}"
        _sample = head_batch[_head][0]
        _index = index_batch[_head][0]
        nnz_voxels = embedded.permute(0, 2, 3, 4, 1)[_sample, _index[0], _index[1], _index[2], :]
        lod_pred_sids = {
            _i: torch.zeros(nnz_voxels.shape[0], dtype=torch.long, device=nnz_voxels.device)
            for _i in range(1, self.max_lod + 1)
        }
        self.top_down_prediction(nnz_voxels, lod_pred_sids, targets_batch, index_batch, head_id=head_id)
        for _lm, _cm in self.test_confusion_matrix.items():
            _lod_pred_sids = lod_pred_sids[int(_lm.split('_')[-1])]
            prediction = self.remaps[_lm].to(_lod_pred_sids.device)[_lod_pred_sids]
            nnz_target = base_batch[_lm][_sample, _index[0], _index[1], _index[2]]
            _cm.add(prediction, nnz_target)
            if self.hparams.predictions_file:
                _full_scene_prediction = torch.zeros_like(base_batch["input"], dtype=torch.long,
                                                          device=base_batch["input"].device).squeeze(1)
                _full_scene_prediction[_sample, _index[0], _index[1], _index[2]] = prediction
                self.predictions.append(tuple(
                    _t.clone().to('cpu') if isinstance(_t, torch.Tensor) else _t
                    for _t in (batch_idx, _lm, _full_scene_prediction,
                               base_batch[_lm][_sample], base_batch['parts'])
                ))

    @staticmethod
    def compute_head_mask(index_batch, child_head_id):
        A = torch.stack(index_batch['head_0'][0])
        B = torch.stack(index_batch[f'head_{child_head_id}'][0])
        matches = (A == B.t().unsqueeze(-1)).all(dim=1).int()
        indices = torch.nonzero(matches)
        return indices[:, 1]

    def top_down_prediction(self, nnz_voxels, lod_pred_sids, targets_batch, index_batch, head_id=0, up_mask=None):
        if up_mask is None:
            subpart_voxels = nnz_voxels
        else:
            subpart_voxels = nnz_voxels[up_mask]
        _logits = self.sm(self.final_layer[head_id](subpart_voxels))
        prediction = _logits.argmax(dim=1)
        if self.hparams.masking == 'gt':
            _target = targets_batch[f'head_{head_id}'][0]
            self.confusion_matrix[head_id].add(prediction, _target)
            cids = _target.unique()
        else:
            cids = prediction.unique()
        head_map = self.top_down_metrics.get_head_id_from_set_id(head_id)
        for _cid in cids:
            _sid, _lod, _c_head_id = head_map[_cid.item()]
            if _lod > self.max_lod:
                return
            if self.hparams.masking == 'gt':
                if _c_head_id is None:
                    mask = up_mask
                else:
                    mask = self.compute_head_mask(index_batch, _c_head_id)
            else:
                if up_mask is None:
                    mask = prediction == _cid
                else:
                    mask = up_mask.clone()
                    mask[mask] = prediction == _cid
            lod_pred_sids[_lod][mask] = _sid
            if _c_head_id is not None:
                self.top_down_prediction(nnz_voxels, lod_pred_sids, targets_batch, index_batch,
                                         head_id=_c_head_id, up_mask=mask)

    def test_end(self, outputs):
        metrics = {}
        if self.hparams.masking == 'gt':
            for _head_id, _confusion_matrix in enumerate(self.confusion_matrix):
                metrics[f'head_{_head_id}'] = _confusion_matrix.compute_iou_metrics()
        for _lm, _confusion_matrix in self.test_confusion_matrix.items():
            metrics[f'{self.hparams.masking}_top_down_to_{_lm}'] = _confusion_matrix.compute_iou_metrics()
        self.write_metrics_file(metrics)
        if self.hparams.predictions_file:
            file_name = self.hparams.predictions_file
            with open(file_name, 'wb+') as f:
                pickle.dump(self.predictions, f)
        return {}

    def generate_test_dataset(self, **kwargs):
        for _lm in self.hparams.test_labels_mapping:
            normalized_transform = MapInstancesToSemanticLabels(
                _lm, mapping_file=kwargs['mapping_file'],
                mapped_key=_lm)
            original_transform = MapInstancesToSemanticLabels(
                _lm, mapping_file=kwargs['mapping_file'], mapped_key=_lm, normalize_labels=False)
            self.remaps[_lm] = torch.tensor(normalized_transform <= original_transform)
            self.test_label_mappings[_lm] = normalized_transform
            self.test_confusion_matrix[_lm] = ConfusionMatrix(
                normalized_transform.num_classes,
                ignore_index=self.ignore_index_for_metric)
        self.test_dataset = generate_test_dataset(**kwargs)
        self.test_dataset.transform.transforms.append(self.labels_to_heads_mapping)
        self.test_dataset.transform.transforms.extend(self.test_label_mappings.values())
        self.top_down_metrics = ComputeHPredictions(self.labels_to_heads_mapping)
        self.max_lod = max(self.hparams.selected_lods)
