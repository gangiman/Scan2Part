from typing import List

import wandb
import torch
import numpy as np
import pandas as pd
import seaborn as sn
import scipy.stats as stats
from matplotlib import pyplot as plt
from pytorch_lightning import Callback

from sklearn import metrics
from sklearn.cluster import MeanShift
from sklearn.metrics import f1_score, precision_score, recall_score

from src.callbacks.wandb_callbacks import get_wandb_logger

sn.set()


class LogInstSegIoU(Callback):
    """Generate f1, precision, recall heatmap every epoch and send it to wandb.
    Expects validation step to return predictions and targets.
    """

    def __init__(
            self,
            class_names: List[str] = None,
            num_classes: int = None,
            bandwidth: float = 1.0,
            num_workers: int = 1,
            plot_3d_points_every_n: bool = False
    ):
        self.metrics = EvaluateInstanceSegmentationPR(num_classes=num_classes)
        self.clustering_method = MeanShift(bandwidth=bandwidth,
                                           n_jobs=num_workers or None)
        self.ready = True
        self.plot_3d_points_every_n = plot_3d_points_every_n

    def on_sanity_check_start(self, trainer, pl_module):
        self.ready = False

    def on_sanity_check_end(self, trainer, pl_module):
        """Start executing this callback only after all validation sanity checks end."""
        self.ready = True

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        """Gather data from single batch."""
        if self.ready:
            for _i, (_embedded, _semantic, _instance_masks) in enumerate(zip(
                    outputs['embedded'], outputs['semantic'], outputs['objects'])):
                pred_instances = self.clustering_method.fit_predict(_embedded.cpu().numpy())
                if self.plot_3d_points_every_n and ((_i * batch_idx) % self.plot_3d_points_every_n == 0):
                    self._make_3d_points_plot(batch, _i, pred_instances, batch_idx)
                self.metrics.update_rates(
                    pred_instances,
                    _semantic.cpu().numpy(),
                    _instance_masks.nonzero()[:, 1].cpu().numpy())

    @staticmethod
    def _make_3d_points_plot(batch, _i, pred_instances, batch_idx):
        batch_coords = batch[0]
        max_instance_id = pred_instances.max()
        colors = plt.cm.jet(pred_instances/max_instance_id, bytes=True)[:, :3]
        xyz = batch_coords[batch_coords[:, 0] == _i, 1:].cpu().numpy()
        pred_point_cloud = np.concatenate([xyz, colors], axis=1)
        wandb.log({f"val/point_cloud_{_i * batch_idx}": wandb.Object3D(pred_point_cloud)})

    def on_validation_epoch_end(self, trainer, pl_module):
        """Generate f1, precision and recall heatmap."""
        if self.ready:
            logger = get_wandb_logger(trainer=trainer)
            experiment = logger.experiment
            df, mAP, mIoU = self.metrics.compute_metrics()
            metrics = {
                'val/mAP': mAP, 'val/mIoU': mIoU,
                'val/PR': wandb.Table(dataframe=df.T)
            }
            # fig = plt.figure(figsize=(12, 12))
            # sns.jointplot(
            #     x="true_ints_count", y="pred_inst_count",
            #     data=pd.DataFrame(
            #         self.metrics.inst_tuples,
            #         columns=("true_ints_count", "pred_inst_count")
            #     ), kind="reg")
            # metrics[f"val/instance_pairs"] = fig
            experiment.log(metrics, commit=False)
            plt.clf()
            self.metrics.reset()


class EvaluateInstanceSegmentationPR:
    def __init__(self, num_classes, ap_threshold=0.50):
        # Init test set wide vars
        self.ap_threshold = ap_threshold
        # total number of classes without background-class
        self.num_classes = num_classes - 1
        self.iou = self.inst_tuples = self.total = self.fps = self.tps = None
        self.reset()

    def reset(self):
        self.inst_tuples = []
        self.total = np.zeros(self.num_classes, dtype=np.int)
        self.fps = [[] for i in range(self.num_classes)]
        self.tps = [[] for i in range(self.num_classes)]
        self.iou = [[] for i in range(self.num_classes)]

    def _distribute_masks_to_classes(self, _instances, _semantic, sizes=None, threshold=0.25):
        _inst_ids = np.unique(_instances)
        num_instances = _inst_ids.shape[0]
        outputs = [[] for i in range(self.num_classes)]
        for gid in _inst_ids:
            # one of the instance masks
            indices = (_instances == gid)
            # predicted semantic class for that instance
            majority_cls = int(stats.mode(_semantic[indices])[0]) - 1
            if sizes is None or (sizes is not None and np.sum(indices) > threshold * sizes[majority_cls]):
                outputs[majority_cls] += [indices]
        return outputs, num_instances

    def update_rates(self, pred_instances, semantic_prediction, truth_inst_masks):
        sizes = dict(zip(*np.unique(semantic_prediction - 1, return_counts=True)))
        proposals, num_proposals = self._distribute_masks_to_classes(
            pred_instances, semantic_prediction, sizes=sizes)
        instances, num_instances = self._distribute_masks_to_classes(
            truth_inst_masks, semantic_prediction)
        self.inst_tuples.append((num_instances, num_proposals))
        for i, (_instances, _proposals) in enumerate(zip(instances, proposals)):
            # for each class
            self.total[i] += len(_instances)
            _num_proposals = len(_proposals)
            # total number of masks per class
            tp = np.zeros(_num_proposals)
            fp = np.zeros(_num_proposals)
            for pid, u in enumerate(_proposals):
                # for each proposal u of class i enumerated by pid
                is_true_positive = False
                for iid, v in enumerate(_instances):
                    iou = (u & v).sum() / (u | v).sum()
                    if iou >= self.ap_threshold:
                        self.iou[i].append(iou)
                        is_true_positive = True
                        break
                if is_true_positive:
                    tp[pid] = 1
                else:
                    fp[pid] = 1
            self.tps[i] += [tp]
            self.fps[i] += [fp]

    def compute_metrics(self):
        p = np.zeros(self.num_classes)
        r = np.zeros(self.num_classes)
        iou = np.zeros(self.num_classes)
        for i in range(self.num_classes):
            tp = np.concatenate(self.tps[i], axis=0)
            fp = np.concatenate(self.fps[i], axis=0)
            iou[i] = np.mean(self.iou[i])
            tp = np.sum(tp)
            fp = np.sum(fp)
            p[i] = tp / (tp + fp)
            r[i] = tp / self.total[i]
        pr_iou = pd.DataFrame({"precision": p, "recall": r, "iou": iou, "num_instances": self.total})
        return pr_iou, np.nanmean(p), np.nanmean(iou)


class LogConfusionMatrixAndMetrics(Callback):
    """Generate confusion matrix every epoch and send it to wandb.
    Expects validation step to return predictions and targets.
    """

    def __init__(self, label_names: str = None):
        self.label_names = pd.read_csv(label_names, squeeze=True, header=None).tolist()
        self.preds = []
        self.targets = []
        self.ready = True
        self.experiment = None

    def on_sanity_check_start(self, trainer, pl_module) -> None:
        self.ready = False

    def on_sanity_check_end(self, trainer, pl_module):
        """Start executing this callback only after all validation sanity checks end."""
        self.ready = True

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        """Gather data from single batch."""
        if self.ready:
            self.preds.extend(outputs['head_logits'][0])
            self.targets.extend(outputs['semantic'])

    def on_validation_epoch_end(self, trainer, pl_module):
        """Generate confusion matrix."""
        if self.ready:
            logger = get_wandb_logger(trainer)
            self.experiment = logger.experiment

            preds = torch.cat(self.preds).cpu().numpy()
            targets = torch.cat(self.targets).cpu().numpy()
            if len(preds.shape) > 1:
                preds = np.argmax(preds, axis=1)
            label_ids = list(range(len(self.label_names)))
            self.plot_confusion_matrix(targets, preds, label_ids)
            self.plot_pr_f1(targets, preds, label_ids)
            self.preds.clear()
            self.targets.clear()

    def plot_confusion_matrix(self, targets, preds, label_ids):
        confusion_matrix = metrics.confusion_matrix(
            y_true=targets,
            y_pred=preds,
            labels=label_ids,
            normalize=True)
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
            yticklabels=self.label_names,
            xticklabels=self.label_names
        )
        # names should be uniqe or else charts from different experiments in wandb will overlap
        self.experiment.log({f"confusion_matrix/{self.experiment.name}": wandb.Image(plt)}, commit=False)
        # according to wandb docs this should also work but it crashes
        # experiment.log(f{"confusion_matrix/{experiment.name}": plt})
        # reset plot
        plt.clf()

    def plot_pr_f1(self, targets, preds, label_ids):
        """Generate f1, precision and recall heatmap."""
        f1 = f1_score(preds, targets, labels=label_ids, average=None)
        r = recall_score(preds, targets, labels=label_ids, average=None)
        p = precision_score(preds, targets, labels=label_ids, average=None)
        self.experiment.log({
            'val/precision': p.mean(),
            'val/recall': r.mean(),
            'val/f1': f1.mean(),
        }, commit=False)
        data = [f1, p, r]
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
            yticklabels=["F1", "Precision", "Recall"],
            xticklabels=self.label_names
        )
        # names should be uniqe or else charts from different experiments in wandb will overlap
        self.experiment.log({f"f1_p_r_heatmap/{self.experiment.name}": wandb.Image(plt)}, commit=False)
        # reset plot
        plt.clf()
