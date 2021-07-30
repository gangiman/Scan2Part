import concurrent
from typing import Optional, Tuple
from pathlib import Path
import pickle
from argparse import Namespace

import pandas as pd
import numpy as np

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split

from tqdm import tqdm
from MinkowskiEngine.utils import batch_sparse_collate
from src.datamodules.datasets.sample_loader import load_sample
from src.datamodules.datasets.scannet import VoxelisedScanNetDataset
from src.datamodules.datasets.scannet import SparseScanNetDataset


def load_vox_and_pkl(_input: Tuple[Path, Path]):
    vox_file, label_file = _input
    _label = pickle.load(label_file.open('rb'))
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
    return load_sample(str(vox_file)).sdf, new_label


def load_sparse_sample(_input: Tuple):
    label_file = _input[0]
    return np.load(label_file.open('rb'),allow_pickle=True).item()


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

    def __init__(self, version='v1', **kwargs):
        super().__init__()
        if version == 'v1':
            self.load_fn = load_vox_and_pkl
            self.dataset_class = VoxelisedScanNetDataset
        elif version == 'v2':
            self.load_fn = load_sparse_sample
            self.dataset_class = SparseScanNetDataset
        else:
            self.load_fn = None
            self.dataset_class = None
        self.hparams = Namespace(**kwargs)
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        self.data = []

    def prepare_data(self):
        """Download data if needed. This method is called only from a single GPU.
        Do not use it to assign state (self.x = y)."""
        pass

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: self.data_train, self.data_val, self.data_test."""
        if stage == 'fit':
            train_df_read = pd.read_csv(self.hparams.train_file, header=None, index_col=False)
            limit = getattr(self.hparams, 'limit', None)
            train_df_read = train_df_read.iloc[:limit]
            val_split_samples = int(np.floor(train_df_read.shape[0] * self.hparams.val_split_ratio))
            num_workers = max(1, self.hparams.num_workers)
            data_files = [
                tuple(self.hparams.data_dir / Path(_path) for _path in _paths)
                for _paths in train_df_read.itertuples(name=None, index=False)]

            with concurrent.futures.ProcessPoolExecutor(num_workers) as executor:
                for _sample in tqdm(executor.map(self.load_fn, data_files), total=len(data_files),
                                    leave=False, desc="Loading data files"):
                    self.data.append(_sample)

            dataset = self.dataset_class(
                self.data, transforms=self.hparams.transforms)

            self.data_train, self.data_val = random_split(
                dataset, (train_df_read.shape[0] - val_split_samples, val_split_samples)
            )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=batch_sparse_collate,
            shuffle=True,
            drop_last=True
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=batch_sparse_collate,
            shuffle=False,
        )
