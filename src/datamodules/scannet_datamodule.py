from typing import Optional

import MinkowskiEngine as ME
from MinkowskiEngine.utils import batch_sparse_collate
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
from sklearn.cluster import MeanShift
import pandas as pd
import numpy as np

#TODO: replace with torchmetrics
# from unet3d.metrics import EvaluateInstanceSegmentationPR


def custom_sparse_collate(*args, **kwargs):
    batch_coords, batch_features, batch_labels = batch_sparse_collate(*args, **kwargs)
    return ME.SparseTensor(batch_features, coordinates=batch_coords, device=batch_features.device), batch_labels


class ScannetDataModule(LightningDataModule):
    """
    Example of LightningDataModule for MNIST dataset.

    A DataModule implements 5 key methods:
        - prepare_data (things to do on 1 GPU/TPU, not on every GPU/TPU in distributed mode)
        - setup (things to do on every accelerator in distributed mode)
        - train_dataloader (the training dataloader)
        - val_dataloader (the validation dataloader(s))
        - test_dataloader (the test dataloader(s))

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
    """

    def __init__(
        self,
        data_dir: str = "data/",
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.kwargs = kwargs
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    @property
    def num_classes(self) -> int:
        return 10

    def prepare_data(self):
        """Download data if needed. This method is called only from a single GPU.
        Do not use it to assign state (self.x = y)."""
        pass

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: self.data_train, self.data_val, self.data_test."""
        if stage == 'fit':
            # datadir = "/data/scannet-voxelized-sdf", testset_split_ratio = 0.1,
            # num_workers = 1, limit = None, instance_mask = "objects", patch_shape = (64, 64, 64),
            # train_file = "/data/scannet-voxelized-sdf/train.csv", val_file = None, sparse = False,
            # ** kwargs
            # self.train_dataset, self.val_dataset = generate_train_and_val_datasets(**merged_kwargs)

            train_df_read = pd.read_csv(self.kwargs.train_file, header=None, index_col=False)
            limit = self.kwargs.get('limit', None)
            train_df_read = train_df_read.iloc[:limit]

            val_split_samples = int(np.floor(train_df_read.shape[0] * self.kwargs.val_split_ratio)

            train_data = read_vox_and_label_files(datadir, train_df, num_workers=num_workers)
            val_data = read_vox_and_label_files(datadir, val_df, num_workers=num_workers)
            train_transforms = [
                RandomFlip(axes=(1, 2)),
                RandomRotate90(axes=(1, 2)),
                RandomRotate(axes=[(2, 1)]),
                ElasticDeformation(execution_probability=0.3)
            ]
            if not sparse:
                train_transforms.append(RandomCrop(patch_shape=patch_shape))
            # return (
                VoxelisedScanNetDataset(train_data, train=True, transforms=train_transforms,
                                        patch_shape=patch_shape, instance_mask=instance_mask,
                                        num_workers=num_workers, sparse=sparse, **kwargs),
                VoxelisedScanNetDataset(val_data, train=False, patch_shape=patch_shape, instance_mask=instance_mask,
                                        num_workers=num_workers, normalize='sample', sparse=sparse,
                                        padding_for_full_conv=16,
                                        **kwargs))


            self.data_train, self.data_val = random_split(
                dataset, (train_df_read.shape[0] - val_split_samples, val_split_samples)
            )
        elif stage == 'test':
            if self.hparams.predictions_file:
                self.predictions = []
            self.test_dataset = generate_test_dataset(**self.hparams.__dict__)
            self.hparams.num_classes = self.test_dataset.num_classes[0]
            self.metrics = EvaluateInstanceSegmentationPR(self.hparams)
            self.clustering_method = MeanShift(bandwidth=self.hparams.bandwidth,
                                               n_jobs=self.hparams.num_workers or None)
        else:
            pass

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=custom_sparse_collate,
            shuffle=True,
            drop_last=True
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=custom_sparse_collate,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=custom_sparse_collate,
            shuffle=False,
        )
