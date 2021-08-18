import json
import k3d
import wandb
import torch
import numpy as np
import pandas as pd
import seaborn as sn
from tqdm import tqdm
import torch.nn.functional as F
from matplotlib import pyplot as plt
from pytorch_lightning import Callback
from collections import defaultdict
from src.callbacks.metric_callbacks import TestingSemSeg
from sklearn import metrics
from sklearn.metrics import f1_score, precision_score, recall_score, jaccard_score

from src.callbacks.wandb_callbacks import get_wandb_logger

sn.set()


class LogHierarchicalTopDown(Callback):
    def __init__(
            self,
            hierarchy_file: str = None
         ):
        with open(hierarchy_file) as fh:
            hierarchy = json.loads(fh.read())
        self.label_names = {}
        self._get_label_names(
            name='ROOT',
            set_id=0,
            children=hierarchy['ROOT']
        )
        self.preds = defaultdict(list)
        self.ready = True
        self.experiment = None

    def _get_label_names(
            self,
            name=None, set_id=None, children=None, lod=1, **kwargs):
        if children is not None and children and len(children) > 1:
            children = sorted(children, key=lambda x: int(x['set_id']))
            self.label_names[f"{lod}_{set_id}"] = [self._get_label_names(**_node, lod=1 + lod) for _node in children]
        return name

    def on_sanity_check_start(self, trainer, pl_module) -> None:
        self.ready = False

    def on_sanity_check_end(self, trainer, pl_module):
        """Start executing this callback only after all validation sanity checks end."""
        self.ready = True

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        """Gather data from single batch."""
        if self.ready:
            for _node_hash, _pred in outputs['head_logits'].items():
                self.preds[_node_hash].append(_pred)

    def on_test_epoch_end(self, trainer, pl_module):
        if self.ready:
            logger = get_wandb_logger(trainer)
            self.experiment = logger.experiment
            for _head_hash, _list_of_preds in self.preds.items():
                logits, norm_targets, targets = list(zip(*_list_of_preds))
                preds = torch.cat(sum(logits, [])).cpu().numpy()
                targets = torch.cat(sum(norm_targets, [])).cpu().numpy()
                if len(preds.shape) > 1:
                    preds = np.argmax(preds, axis=1)
                label_names = self.label_names[_head_hash]
                label_ids = list(range(len(label_names)))
                self.plot_confusion_matrix(targets, preds, label_ids, _head_hash, label_names)
                self.plot_pr_f1(targets, preds, label_ids, _head_hash, label_names)
            self.preds.clear()

    def plot_confusion_matrix(self, targets, preds, label_ids, head_id, label_names, mode='test'):
        confusion_matrix = metrics.confusion_matrix(
            y_true=targets,
            y_pred=preds,
            labels=label_ids)
        # set figure size
        plt.figure(figsize=(14, 12))
        # set labels size
        sn.set(font_scale=1.4)
        # set font size
        sn.heatmap(
            confusion_matrix,
            annot=True,
            annot_kws={"size": 8},
            fmt="g",
            yticklabels=label_names,
            xticklabels=label_names
        )
        # names should be uniqe or else charts from different experiments in wandb will overlap
        self.experiment.log({f"{mode}/head_{head_id}/confusion_matrix": wandb.Image(plt)}, commit=False)
        # according to wandb docs this should also work but it crashes
        # experiment.log(f{"confusion_matrix/{experiment.name}": plt})
        # reset plot
        plt.clf()

    def plot_pr_f1(self, targets, preds, label_ids, head_id, label_names, mode='test'):
        """Generate f1, precision and recall heatmap."""
        f1 = f1_score(preds, targets, labels=label_ids, average=None)
        r = recall_score(preds, targets, labels=label_ids, average=None)
        p = precision_score(preds, targets, labels=label_ids, average=None)
        iou = jaccard_score(preds, targets, labels=label_ids, average=None)
        data = [f1, p, r, iou]
        df = pd.DataFrame({'f1_score': f1, 'precision': p, 'recall': r, 'iou': iou}, index=label_names)
        self.experiment.log({
            # f'{mode}/head_{self.head_id}/precision': p.mean(),
            # f'{mode}/head_{self.head_id}/recall': r.mean(),
            # f'{mode}/head_{self.head_id}/f1': f1.mean(),
            f'{mode}/head_{head_id}/mIoU': iou.mean(),
            f'{mode}/head_{head_id}/table': wandb.Table(dataframe=df)
        }, commit=False)
        # set figure size
        plt.figure(figsize=(14, 8))
        # set labels size
        sn.set(font_scale=1.2)
        # set font size
        sn.heatmap(
            data,
            annot=True,
            annot_kws={"size": 10},
            fmt=".3f",
            yticklabels=["F1", "Precision", "Recall", "IoU"],
            xticklabels=label_names
        )
        # names should be uniqe or else charts from different experiments in wandb will overlap
        self.experiment.log({f"{mode}/head_{head_id}/f1_p_r_heatmap": wandb.Image(plt)}, commit=False)
        # reset plot
        plt.clf()


id_mappings = {
    '2_to_1': {0: [0, 34],
             1: [23, 27],
             2: [17, 8],
             3: [1, 20],
             4: [2, 16, 25],
             5: [3, 18],
             6: [4, 12, 31],
             7: [5, 6, 19],
             8: [7, 11, 13, 22, 28],
             9: [9, 30],
             10: [10, 14, 21, 26, 32],
             11: [15, 24, 33],
             12: [29, 35]},
    '3_to_2': {0: [12, 30, 53], 23: [35, 39], 27: [41, 54], 17: [1], 8: [22, 48], 1: [2, 4, 61], 20: [29, 31],
     16: [75], 25: [40, 60], 3: [3, 6, 7, 9, 14, 43, 44, 49, 59], 18: [70], 4: [5], 12: [15], 31: [68],
     5: [8, 65], 6: [10, 23], 19: [27], 7: [11, 21, 24, 37, 45], 11: [13, 18, 20, 25, 33, 38, 64, 66, 71],
     13: [16, 19, 47, 77], 22: [34, 36, 50, 58, 74], 28: [51], 9: [72, 73], 30: [52], 10: [26, 57],
     14: [17, 42, 55, 67, 76], 21: [32, 69], 26: [56], 32: [62], 15: [28], 33: [63, 0], 29: [46], 35: [78]}}


def project_probs(probs, new_dim, type='3_to_2'):
    output = []
    for prob in probs:
        new_prob = torch.zeros((prob.shape[0], new_dim), device=prob.device)
        for _id, _lod_3_ids in id_mappings[type].items():
            new_prob[:, _id] = prob[:, _lod_3_ids].sum(dim=1)
        output.append(new_prob)
    return output


class BottomUpSemSeg(TestingSemSeg):
    def __init__(
            self,
            project_to='lod_2',
            from_head_id=0,
            **kwargs):
        super().__init__(**kwargs)
        self.project_to = project_to
        self.from_head_id = from_head_id

    def project_logits(self, logits):
        probs = [F.softmax(_logit, dim=1) for _logit in logits]
        proj_logits = project_probs(probs, 36, type='3_to_2')
        if self.project_to == 'lod_2':
            return proj_logits
        elif self.project_to == 'lod_1':
            return project_probs(proj_logits, 13, type='2_to_1')

    def on_test_batch_end(
            self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        if self.ready:
            batch_coords = batch[0]
            self.coords.extend([
                batch_coords[batch_coords[:, 0] == _i, 1:]
                for _i in range(batch_coords[:, 0].max() + 1)
            ])
            top_logits = self.project_logits(outputs['head_logits'][self.from_head_id])
            self.preds.extend(top_logits)
            self.targets.extend(outputs[self.semantic_key])

