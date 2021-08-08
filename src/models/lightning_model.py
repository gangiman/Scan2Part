from argparse import Namespace
import omegaconf
from collections import defaultdict
import MinkowskiEngine as ME

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from pytorch_lightning import LightningModule
from src.models.sparse.res16unet import Res16UNet18A
from src.models.sparse.res16unet import Res16UNet34C, Res16UNet34B
from src.utils.poly_lr_decay import PolynomialLRDecay
from src.submanifold.unet import SubmanifoldUNet


def stack_instance_dicts(test_list):
    res = defaultdict(list)
    for sub in test_list:
        for key in sub:
            res[key].append(sub[key])
    return res


class Residual3DUnet(LightningModule):
    def __init__(self,
                 sparse_backbone_type='Res16UNet34C',
                 in_channels=1,
                 f_maps=32,
                 conv1_kernel_size=3,
                 **kwargs):
        super().__init__()
        self.save_hyperparameters()

        if sparse_backbone_type == 'Res16UNet34C':
            self.model = Res16UNet34C(
                in_channels, f_maps,
                Namespace(bn_momentum=0.05, conv1_kernel_size=conv1_kernel_size))
        elif sparse_backbone_type == 'Res16UNet18A':
            self.model = Res16UNet18A(
                in_channels, f_maps,
                Namespace(bn_momentum=0.05, conv1_kernel_size=conv1_kernel_size))
        elif sparse_backbone_type == 'Res16UNet34B':
            self.model = Res16UNet34B(
                in_channels, f_maps,
                Namespace(bn_momentum=0.05, conv1_kernel_size=conv1_kernel_size))
        #####################################################################################
        elif sparse_backbone_type == 'SubmanifoldUNet':
            self.model = SubmanifoldUNet()
        #####################################################################################
        else:
            raise AssertionError(f"Unknown backbone type {sparse_backbone_type}")

    def forward(self, batch):
        batch_coords, batch_features, batch_labels = batch
        batch_size = len(batch_labels)
        dict_of_lists = stack_instance_dicts(batch_labels)
        #####################################################################################
        if self.hparams.minkowski:
            features = ME.SparseTensor(
                batch_features,
                coordinates=batch_coords,
                device=batch_features.device)
            sparse_embedded = self.model(features)
            embedded = [sparse_embedded.features_at(i) for i in range(batch_size)]
        else:
            embedded = self.model([batch_coords, batch_features])
        #####################################################################################    
        return embedded, dict_of_lists

    def configure_optimizers(self):
        if self.hparams.optimizer == 'SGD':
            optimizer = optim.SGD(
                self.parameters(), lr=self.hparams.learning_rate, momentum=self.hparams.sgd_momentum,
                dampening=self.hparams.sgd_dampening, weight_decay=self.hparams.weight_decay)
        else:
            optimizer = optim.Adam(
                self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay,
                betas=(self.hparams.adam_beta1, self.hparams.adam_beta2))

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
            elif isinstance(self.hparams.milestones, omegaconf.listconfig.ListConfig):
                milestones = omegaconf.OmegaConf.to_container(self.hparams.milestones)
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
