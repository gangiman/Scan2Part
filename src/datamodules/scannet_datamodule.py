import concurrent
from typing import Optional, Tuple
from pathlib import Path
import pickle
from argparse import Namespace

import pandas as pd
import numpy as np
from sklearn.cluster import MeanShift

import MinkowskiEngine as ME
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split

from tqdm import tqdm
from MinkowskiEngine.utils import batch_sparse_collate

from src.datamodules.transforms import ToTensor
from src.datamodules.transforms import MapInstancesToSemanticLabels
from src.datamodules.transforms import ComputeInstanceMasks
from src.datamodules.transforms import RandomCrop
from src.datamodules.transforms import RandomFlip
from src.datamodules.transforms import RandomRotate90
from src.datamodules.transforms import RandomRotate
from src.datamodules.transforms import ElasticDeformation
from src.datamodules.transforms import ToSparse
from src.datamodules.transforms import NormalizeInstanceLabels
from src.datamodules.transforms import GetInstanceMaskForLoD


from src.datamodules.datasets.sample_loader import load_sample
from src.datamodules.datasets.scannet import VoxelisedScanNetDataset
from src.datamodules.transforms import RandomFlip
from src.datamodules.transforms import RandomRotate90
from src.datamodules.transforms import RandomRotate
from src.datamodules.transforms import ElasticDeformation

#TODO: replace with torchmetrics
# from unet3d.metrics import EvaluateInstanceSegmentationPR


def get_data(_input: Tuple[Path, Path]):
    vox_file, label_file = _input
    _label = pickle.load(label_file.open())
    if 'object' in _label and np.all(_label['object'] < 0):
        return None, None
    new_label = {}
    for label_key, new_key in [('mask', 'parts'), ('object', 'objects')]:
        if label_key not in _label:
            continue
        if _label[label_key].ndim == 4:
            new_label[new_key] = _label[label_key][0].astype(np.int)
        else:
            new_label[new_key] = _label[label_key].astype(np.int)
    return load_sample(vox_file).sdf, new_label


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

    def __init__(self, **kwargs):
        super().__init__()
        self.hparams = Namespace(**kwargs)
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

            train_df_read = pd.read_csv(self.hparams.train_file, header=None, index_col=False)
            limit = getattr(self.hparams, 'limit', None)
            train_df_read = train_df_read.iloc[:limit]
            val_split_samples = int(np.floor(train_df_read.shape[0] * self.hparams.val_split_ratio))
            # train_data = read_vox_and_label_files(self.data_dir, train_df_read, num_workers=self.hparams.num_workers)
            # def read_vox_and_label_files(datadir, paths_df, num_workers=1):
            num_workers = max(1, self.hparams.num_workers)
            data_files = [
                (self.hparams.data_dir / Path(_vox), self.hparams.data_dir / Path(_label))
                for _vox, _label in train_df_read.itertuples(name=None, index=False)]
            inputs, labels = [], []
            with concurrent.futures.ProcessPoolExecutor(num_workers) as executor:
                # ind = 1 # for debug
                for _vox, _label in tqdm(executor.map(get_data, data_files), total=len(data_files),
                                         leave=False, desc="Loading data files"):
                    # print(ind) # for debug
                    # ind += 1 # for debug
                    inputs.append(_vox)
                    labels.append(_label)
                
            train_transforms = [
                RandomFlip(axes=(1, 2)),
                RandomRotate90(axes=(1, 2)),
                RandomRotate(axes=[(2, 1)]),
                ElasticDeformation(execution_probability=0.3),
                ToTensor(expand_dims=False)
            ]

            # return (
            dataset = VoxelisedScanNetDataset(
                inputs, labels, transforms=train_transforms)

            self.data_train, self.data_val = random_split(
                dataset, (train_df_read.shape[0] - val_split_samples, val_split_samples)
            )
        # elif stage == 'test':
        #     if grid_step is None:
        #         padding_for_full_conv = 8
        #         normalize = 'sample'
        #     else:
        #         padding_for_full_conv = None
        #         normalize = 'dataset'
        #     test_df = pd.read_csv(test_file, header=None, index_col=False)
        #     if limit is not None:
        #         assert isinstance(limit, int) and limit > 0
        #         test_df = test_df.iloc[:limit]
        #     inputs, labels = read_vox_and_label_files(datadir, test_df, num_workers=num_workers)
        #     self.data_test = VoxelisedScanNetDataset(
        #         (inputs, labels), num_workers=num_workers, grid_step=grid_step,
        #        dataset_multiplier=None, one_hot_encode_instances=False,
        #        padding_for_full_conv=padding_for_full_conv,
        #        normalize=normalize, **kwargs)

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=custom_sparse_collate,
            shuffle=True,
            drop_last=True
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=custom_sparse_collate,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=custom_sparse_collate,
            shuffle=False,
        )
