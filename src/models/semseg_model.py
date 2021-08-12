from typing import Dict, Optional, List

import torch
from torch import nn
import numpy as np
from src.models.lightning_model import Residual3DUnet
import sparseconvnet as scn


class SemanticHeadLoss(nn.Module):
    def __init__(
            self,
            num_classes: int = None,
            loss_weight: float = 1.0,
            semantic_key: str = 'semantic',
            weight_mode: str = 'median',
            class_weights_file: Optional[str] = None,
            f_maps: int = 32,
            ##########################
            minkowski: bool = False,
            ##########################
            **kwargs
    ):
        super().__init__()
        if class_weights_file is None:
            weights = None
        else:
            class_counts = torch.tensor(np.load(class_weights_file))
            weights = getattr(class_counts, weight_mode)() / class_counts.to(torch.float)
        self.minkowski = minkowski
        self.loss_weight = loss_weight
        self.semantic_key = semantic_key
        self.final_layer = nn.Linear(f_maps, num_classes, bias=False)
        self.criterion = nn.CrossEntropyLoss(weight=weights)

    def forward(self, features: List[torch.Tensor], batch: Dict[str, List[torch.Tensor]]):
        ############################################################
        if self.minkowski:
            logits = [self.final_layer(_features) for _features in features]
        else:
            logits = [self.final_layer(_features) for _features in features]
#             logits = self.final_layer(features)
        target = batch[self.semantic_key]

#         print('\n#############################')
#         print(len(logits)) # 26861
#         print(logits[0].shape) # 13
#         print(len(target)) # 8
#         print(target[0].shape) # 3202
#         print(sum([len(el) for el in target])) # 26861
#         print('#############################\n')
        ############################################################
        if self.minkowski:
            return self.loss_weight * torch.stack([
                self.criterion(_logits, _target)
                for _logits, _target in zip(logits, target)
            ]).sum(), logits
        else:
            ################################################################
#             target = torch.cat(target, 0)
#             print('\n#############################')
#             print('forward:')
#             print(len(target))
#             print(len(logits))
#             print('#############################\n')
#             return self.loss_weight * self.criterion(logits, target), logits

            return self.loss_weight * torch.stack([
                self.criterion(_logits, _target)
                for _logits, _target in zip(logits, target)
            ]).sum(), logits
            ################################################################

class SemanticSegmentation(Residual3DUnet):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.heads = nn.ModuleList([
            SemanticHeadLoss(**_head, f_maps=self.hparams.f_maps, minkowski=self.hparams.minkowski)
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
    
    #######################################################
    def training_epoch_end(self, outputs) -> None:
        if not self.hparams.minkowski:
            scn.forward_pass_multiplyAdd_count = 0
            scn.forward_pass_hidden_states = 0
            
    def validation_epoch_end(self, outputs) -> None:
        if not self.hparams.minkowski:
            scn.forward_pass_multiplyAdd_count = 0
            scn.forward_pass_hidden_states = 0
    #######################################################

