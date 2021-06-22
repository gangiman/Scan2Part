from argparse import Namespace

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from torch.utils.data import DataLoader

import pytorch_lightning as pl

import MinkowskiEngine as ME
from models.res16unet import Res16UNet14
from models.res16unet import Res16UNet18A, Res16UNet18B, Res16UNet18D, Res16UNet18
from models.res16unet import Res16UNet34C, Res16UNet34B, Res16UNet34A, Res16UNet34
from unet3d.model import ResidualUNet3D
from datasets.scannet import generate_train_and_val_datasets
from MinkowskiEngine.utils import batch_sparse_collate
from utils.poly_lr_decay import PolynomialLRDecay


def custom_sparse_collate(*args, **kwargs):
    batch_coords, batch_features, batch_labels = batch_sparse_collate(*args, **kwargs)
    return ME.SparseTensor(batch_features, coordinates=batch_coords, device=0), batch_labels
#     return ME.SparseTensor(batch_features, coordinates=batch_coords), batch_labels

# 
class Residual3DUnet(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.train_dataset = None
        self.val_dataset = None
        if self.hparams.sparse:
            if self.hparams.sparse_backbone_type == 'Res16UNet34C':
                self.model = Res16UNet34C(self.hparams.in_channels, self.hparams.f_maps,
                                        Namespace(bn_momentum=0.05,
                                        conv1_kernel_size=self.hparams.conv1_kernel_size))

            elif self.hparams.sparse_backbone_type == 'Res16UNet18A':
                self.model = Res16UNet18A(self.hparams.in_channels, self.hparams.f_maps,
                                        Namespace(bn_momentum=0.05,
                                        conv1_kernel_size=self.hparams.conv1_kernel_size))

            elif self.hparams.sparse_backbone_type == 'Res16UNet34B':
                self.model = Res16UNet34B(self.hparams.in_channels, self.hparams.f_maps,
                                        Namespace(bn_momentum=0.05,
                                        conv1_kernel_size=self.hparams.conv1_kernel_size))

            if self.hparams.instance_mask == 'semantic':
                self.collate_fn = custom_sparse_collate
                # self.collate_fn = None
            if self.hparams.instance_mask == 'objects':
                # self.collate_fn = None
                self.collate_fn = custom_sparse_collate # for instance debug
        else:
            self.model = ResidualUNet3D(**self.hparams.__dict__)
            self.collate_fn = None

    def configure_optimizers(self):
        if self.hparams.optimizer == 'SGD':
            optimizer = optim.SGD(self.parameters(), lr=self.hparams.learning_rate, momentum=self.hparams.sgd_momentum,
                                dampening=self.hparams.sgd_dampening, weight_decay=self.hparams.weight_decay)
        else:
            optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay,
                                betas=(self.hparams.adam_beta1, self.hparams.adam_beta2),)

        if self.hparams.lr_scheduler == 'CosineAnnealingWithRestartsLR':
            scheduler = CosineAnnealingWarmRestarts(
                optimizer, T_0=self.hparams.T_0, T_mult=self.hparams.T_mult, eta_min=self.hparams.eta_min)
        elif self.hparams.lr_scheduler == 'OneCycleLR':
            # https://pytorch.org/docs/stable/optim.html#torch.optim.lr_scheduler.OneCycleLR
            # steps_per_epoch = len(data_loader)
            scheduler = OneCycleLR(
                optimizer, max_lr=0.01, steps_per_epoch=self.hparams.steps_per_epoch, epochs=self.hparams.max_epochs)
        elif self.hparams.lr_scheduler == 'MultiStepLR':
            if isinstance(self.hparams.milestones, torch.Tensor):
                milestones = self.hparams.milestones.tolist()
            else:
                milestones = self.hparams.milestones
            scheduler = MultiStepLR(
                optimizer, gamma=self.hparams.gamma,
                milestones=milestones)
        elif self.hparams.lr_scheduler == 'PolynomialLR':
            scheduler = PolynomialLRDecay(
                optimizer, max_decay_steps=self.hparams.max_decay_steps,
                end_learning_rate=self.hparams.end_learning_rate, power=self.hparams.polynomial_lr_power)
        else:
            raise AssertionError(f"Unknown LR Scheduler {self.hparams.lr_scheduler}.")
        return [optimizer], [scheduler]

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size, collate_fn=self.collate_fn,
                          shuffle=self.hparams.shuffle, num_workers=self.hparams.num_workers, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, collate_fn=self.collate_fn, batch_size=1)

    def generate_train_and_val_datasets(self, **kwargs):
        merged_kwargs = {**self.hparams, **kwargs}
        self.train_dataset, self.val_dataset = generate_train_and_val_datasets(**merged_kwargs)
