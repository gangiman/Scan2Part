from typing import List

import k3d
import wandb
import torch
import numpy as np
import pandas as pd
import seaborn as sn
from matplotlib import pyplot as plt
from pytorch_lightning import Callback

from sklearn import metrics
from sklearn.cluster import MeanShift
from sklearn.metrics import f1_score, precision_score, recall_score

from src.callbacks.wandb_callbacks import get_wandb_logger
from src.utils.metrics import EvaluateInstanceSegmentationPR
from src.utils.plotting import plot_3d_voxels_as_k3d_html

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
                    outputs['embedded'], outputs['semantic'], outputs['object'])):
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


class LogConfusionMatrixAndMetrics(Callback):
    """Generate confusion matrix every epoch and send it to wandb.
    Expects validation step to return predictions and targets.
    """

    def __init__(self, label_names: str = None, plot_3d_points_every_n: bool = False):
        self.label_names = pd.read_csv(label_names, squeeze=True, header=None).tolist()
        self.plot_3d_points_every_n = plot_3d_points_every_n
        self.preds = []
        self.targets = []
        self.coords = []
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
            batch_coords = batch[0]
            self.coords.extend([
                batch_coords[batch_coords[:, 0] == _i, 1:]
                for _i in range(batch_coords[:, 0].max() + 1)
            ])
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
            for _i, (_coord, _target, _pred) in enumerate(zip(self.coords, self.targets, self.preds)):
                if self.plot_3d_points_every_n and (_i % self.plot_3d_points_every_n == 0):
                    html = plot_3d_voxels_as_k3d_html(_coord, _pred)
                    self.experiment.log({f"val/k3d_voxels_{_i}": wandb.Html(html)})
            self.coords.clear()
            self.preds.clear()
            self.targets.clear()

    def plot_confusion_matrix(self, targets, preds, label_ids):
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
