import pickle
from typing import Dict, Optional, List

import torch
from torch import nn
import numpy as np
from src.models.lightning_model import Residual3DUnet


class SemanticHeadLoss(nn.Module):
    def __init__(
            self,
            num_classes: int = None,
            loss_weight: float = 1.0,
            semantic_key: str = 'semantic',
            weight_mode: str = 'median',
            class_weights_file: Optional[str] = None,
            f_maps: int = 32
    ):
        super().__init__()
        if class_weights_file is None:
            weights = None
        else:
            class_counts = torch.tensor(np.load(class_weights_file))
            weights = getattr(class_counts, weight_mode)() / class_counts.to(torch.float)
        self.loss_weight = loss_weight
        self.semantic_key = semantic_key
        self.final_layer = nn.Linear(f_maps, num_classes, bias=False)
        self.criterion = nn.CrossEntropyLoss(weight=weights)

    def forward(self, features: List[torch.Tensor], batch: Dict[str, List[torch.Tensor]]):
        target = batch[self.semantic_key]
        return self.loss_weight * torch.stack([
            self.criterion(self.final_layer(_features), _target)
            for _features, _target in zip(features, target)
        ]).sum()


class SemanticSegmentation(Residual3DUnet):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.loss = nn.ModuleList()
        self.final_layer = nn.ModuleList()
        self.loss_weights = []

        self.heads = nn.ModuleList([
            SemanticHeadLoss(**_head, f_maps=self.hparams.f_maps)
            for _head in self.hparams.heads])

    def shared_step(self, batch):
        embedded, dict_of_lists = self.forward(batch)
        loss = torch.stack([_head.forward(embedded, dict_of_lists) for _head in self.heads]).sum()
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
        accumulated = None
        embedded = self.model(batch["input"])
        _predictions = {'scene_id': batch_idx, 'parts': batch['parts']}
        for _label_mapping, _final_layer in self.final_layer.items():
            _predictions[_label_mapping] = {'gt': batch[f'semantic_{_label_mapping}']}
            logits = _final_layer(embedded.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)
            logits = self.log_softmax(logits)
            _predictions[_label_mapping]["pred"] = logits.argmax(dim=1)
            if self.hparams.accumulate_logits:
                if accumulated is not None:
                    logits += accumulated[:, self.expansions[_label_mapping], ...]
                    _predictions[_label_mapping]["accumulated"] = logits.argmax(dim=1)
                accumulated = logits
            _cm = self.confusion_matrix[_label_mapping]
            _cm.add(logits, batch[f'semantic_{_label_mapping}'])
        for _from_lm, _d in self.bottom_up_cms.items():
            for _to_lm, _test_cm in _d.items():
                for _type, _pred in _predictions[_from_lm].items():
                    if _type != 'gt':
                        _predictions[_to_lm][f"bottom_up_from_{_from_lm}_{_type}"] = \
                            torch.tensor(_test_cm.project_predicted, device=_pred.device)[_pred]
                    if _type == 'pred':
                        _test_cm.add(_pred, batch[f'semantic_{_to_lm}'])
        if self.predictions is not None:
            self.predictions.append(to_cpu(_predictions))
        else:
            del _predictions
        return {}

    def test_epoch_end(self, outputs):
        metrics = {}
        for _lm, _confusion_matrix in self.confusion_matrix.items():
            metrics[f'head_{_lm}'] = _confusion_matrix.compute_iou_metrics()
        for _from_lm, _d in self.bottom_up_cms.items():
            for _to_lm, _test_cm in _d.items():
                metrics[f'proj_{_from_lm}_to_{_to_lm}'] = _test_cm.compute_iou_metrics()
        write_yaml(self.hparams.metrics_file, metrics)
        if self.hparams.predictions_file:
            file_name = self.hparams.predictions_file
            with open(file_name, 'wb+') as f:
                pickle.dump(self.predictions, f)
        return {}
