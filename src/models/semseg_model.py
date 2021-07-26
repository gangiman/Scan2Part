from typing import Dict, Optional, List

import torch
from torch import nn
import numpy as np
import pandas as pd
from src.models.lightning_model import Residual3DUnet


class SemanticHeadLoss(nn.Module):
    def __init__(
            self,
            num_classes: int = None,
            loss_weight: float = 1.0,
            semantic_key: str = 'semantic',
            weight_mode: str = 'median',
            class_weights_file: Optional[str] = None,
            f_maps: int = 32,
            **kwargs
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
        logits = [self.final_layer(_features) for _features in features]
        target = batch[self.semantic_key]
        return self.loss_weight * torch.stack([
            self.criterion(_logits, _target)
            for _logits, _target in zip(logits, target)
        ]).sum(), logits


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
        loss_terms = []
        head_logits = []
        for _head in self.heads:
            loss, logits = _head.forward(embedded, dict_of_lists)
            loss_terms.append(loss)
            head_logits.append(logits)
        return torch.stack(loss_terms).sum(), head_logits, dict_of_lists

    def training_step(self, batch, batch_idx):
        loss, *_ = self.shared_step(batch)
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, head_logits, masks_dict = self.shared_step(batch)
        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {'head_logits': head_logits, **masks_dict}

