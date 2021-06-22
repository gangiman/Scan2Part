import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torchvision.transforms import Compose
from torch.utils.data import Dataset, DataLoader
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


def read_labels(path_to_file):
    with open(path_to_file, 'rb') as fh:
        pkl = pickle.load(fh)
    return pkl


def get_data(_input):
    vox_file, label_file = _input
    _label = read_labels(label_file)
    if 'object' in _label and np.all(_label['object'] < 0):
        return None, None
    new_label = {}
    for label_key, new_key in [('mask','parts'),('object', 'objects')]:
        if label_key not in _label:
            continue
        if _label[label_key].ndim == 4:
            new_label[new_key] = _label[label_key][0].astype(np.int)
        else:
            new_label[new_key] = _label[label_key].astype(np.int)
    return load_sample(vox_file).sdf, new_label


def read_vox_and_label_files(datadir, paths_df, num_workers=1):
    num_workers = max(1, num_workers)
    extend = partial(os.path.join, datadir)
    data_files = [
        (extend(_vox), extend(_label))
        for _vox, _label in paths_df.itertuples(name=None, index=False)]
    inputs, labels = [], []
    with concurrent.futures.ProcessPoolExecutor(num_workers) as executor:
        # ind = 1 # for debug
        for _vox, _label in tqdm(executor.map(get_data, data_files), total=len(data_files),
                                 leave=False, desc="Loading data files"):
            # print(ind) # for debug
            # ind += 1 # for debug
            inputs.append(_vox)
            labels.append(_label)
    return inputs, labels


class VoxelisedScanNetDataset(Dataset):
    def __init__(self, data, train=False,
                 transforms=None,
                 patch_shape=(64, 64, 64),
                 normalize='dataset',
                 num_workers: int = 1,
                 instance_mask="objects",
                 labels_mapping=None,
                 mapping_file=None,
                 padding_for_full_conv=None,
                 lod: int = 1,
                 one_hot_encode_instances=True,
                 sparse=False,
                 **kwargs):
        """
        Create dataset of voxels, optionally cropped
        :param data: pair of lists with numpy tensors for input and labels
        :param dataset_multiplier: for random cropping specify how many crops are taken from one scene
        :param patch_shape: crop size, tuple of length 3
        :param normalize: when set to 'dataset' normalisation performed dataset-wise  if 'sample' per-sample
        :param grid_step: step of crops along all 3 dims
        :param num_workers: number of workers for concurrent crop generation
        :param instance_mask: can be ('semantic', 'parts', 'objects') for different tasks
        :param labels_mapping: name of part_ids to semantic labels mapping
        :param mapping_file: path to file where `labels_mapping` can be found
        :param padding_for_full_conv: make all dimensions of tensors divisible by `padding_for_full_conv`, is needed
            for fully-convolutional processing of tensors
        :param kwargs:
        """
        self.scan_vox, self.labels = data
        self.train = train
        self.num_scenes = len(self.labels)
        self.patch_shape = np.asarray(patch_shape, dtype=np.int)
        self.num_workers = max(num_workers, 1)
        self.instance_mask = instance_mask
        self.transform = None
        self.scenes_map = None
        self.slices = None
        if sparse:
            self.collate_fn = batch_sparse_collate
            # self.collate_fn = None
        else:
            self.collate_fn = None
        self.num_classes = []
        self.label_mappings = {}
        self._mapping_file = mapping_file
        if transforms is None:
            transforms = []
        transforms.append(ToTensor(expand_dims=False))
        if not sparse:
            self._pad_input_tensors()
        if isinstance(labels_mapping, str):
            labels_mapping = (labels_mapping,)
        label_key = None
        if labels_mapping is not None:
            assert labels_mapping
            for _lm in labels_mapping:
                label_key = f"semantic_{_lm}"
                part_label_mapping = MapInstancesToSemanticLabels(
                    labels_mapping=_lm, mapping_file=mapping_file, mapped_key=label_key, sparse=sparse)
                self.num_classes.append(part_label_mapping.num_classes)
                self.label_mappings[_lm] = part_label_mapping
                transforms.append(part_label_mapping)

        self.transform = Compose(transforms)
        self.num_samples = self.num_scenes
        if self.instance_mask != 'semantic':
            if (self.instance_mask == 'objects' and lod == 1) or self.instance_mask == 'parts':
                self.transform.transforms.append(NormalizeInstanceLabels(self.instance_mask, label_key))
            else:
                self.transform.transforms.append(GetInstanceMaskForLoD(labels_mapping[0]))
            if one_hot_encode_instances:
                max_num_of_instances_in_dataset = self._compute_maximum_number_of_instances()
                self.transform.transforms.append(
                    ComputeInstanceMasks(max_num_of_instances_in_dataset, self.instance_mask))
        if sparse:
            self.transform.transforms.append(ToSparse(label_key=label_key, bg_value=0,
                                        instance_mask=instance_mask))


    def _compute_maximum_number_of_instances(self):
        _key = self.instance_mask + "_size"
        # print(self[0].keys())
        return torch.stack([self[_ind][_key] for _ind in range(self.num_samples)]).max()

    def __getitem__(self, item):
        sample = {'input': self.scan_vox[item]}
        sample.update(self.labels[item])
        if self.transform is not None:
            return self.transform(sample)
        else:
            return sample

    def __len__(self):
        return self.num_samples

    def compute_weights(self, num_classes, weights_mode, ignore_index=None, mapped_key='semantic', sparse=False):
        class_counts = torch.zeros(num_classes, dtype=torch.long)
        if sparse:
            for *_, labels in tqdm(
                    DataLoader(self, batch_size=self.num_workers * 4,
                            num_workers=self.num_workers, collate_fn=self.collate_fn),
                    leave=False, unit='batch', desc="Counting classes for weights"):
                # print('I AM HERE')
                _ind, _counts = (labels - 1).unique(return_counts=True) # (labels - 1) from 0 to 19
                class_counts[_ind] += _counts
        else:
            full_labels = [
                batch[mapped_key]
                for batch in tqdm(DataLoader(self, batch_size=self.num_workers * 4, num_workers=self.num_workers),
                                leave=False, unit='batch', desc="Counting classes for weights")
            ]
            _ind, _counts = torch.cat(full_labels, dim=0).unique(return_counts=True)
            class_counts[_ind] += _counts

        class_counts[class_counts == 0] = 1        
        slice_from = int(ignore_index == 0)
        class_counts = class_counts[slice_from:] # len = 20
        weights = getattr(class_counts[slice_from:], weights_mode)() / class_counts.to(torch.float)
        return weights, class_counts

    def _pad_input_tensors(self):
        for i in tqdm(range(self.num_scenes), leave=False,
                      desc=f"Padding scene tensors to be at least of size {self.patch_shape}"):
            if any(ss < ps for ss, ps in zip(self.scan_vox[i].shape[1:], self.patch_shape)):
                pads = [[0, 0]]
                for _dim in range(3):
                    shape_diff = self.scan_vox[i].shape[_dim + 1] - self.patch_shape[_dim]
                    if shape_diff < 0:
                        shape_diff *= -1
                        if shape_diff % 2 == 0:
                            pads.append([shape_diff // 2, shape_diff // 2])
                        else:
                            pads.append([shape_diff // 2, shape_diff // 2 + 1])
                    else:
                        pads.append([0, 0])
                self.scan_vox[i] = np.pad(self.scan_vox[i], pads, mode='edge')
                for _key in ('parts', 'objects'):
                    if _key in self.labels[i]:
                        self.labels[i][_key] = np.pad(
                            self.labels[i][_key], pads[1:], mode='constant', constant_values=[(-2, -2)]*3)


def generate_train_and_val_datasets(datadir="/data/scannet-voxelized-sdf", testset_split_ratio=0.1,
                                    num_workers=1, limit=None, instance_mask="objects", patch_shape=(64, 64, 64),
                                    train_file="/data/scannet-voxelized-sdf/train.csv", val_file=None, sparse=False,
                                    **kwargs):
    if not isinstance(patch_shape, (list, tuple)):
        patch_shape = patch_shape.tolist()
    train_df_read = pd.read_csv(train_file, header=None, index_col=False)
    if limit is not None:
        assert isinstance(limit, int) and limit > 0
        train_df_read = train_df_read.iloc[:limit]
    if val_file is None:
        val_samples_multiplier = int(np.floor(train_df_read.shape[0] * testset_split_ratio))
        val_df = train_df_read.iloc[:val_samples_multiplier]
        train_df = train_df_read.iloc[val_samples_multiplier:]
    else:
        train_df = train_df_read
        val_df = pd.read_csv(val_file, header=None, index_col=False)
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
    return (
        VoxelisedScanNetDataset(train_data, train=True, transforms=train_transforms,
                                patch_shape=patch_shape, instance_mask=instance_mask,
                                num_workers=num_workers, sparse=sparse, **kwargs),
        VoxelisedScanNetDataset(val_data, train=False, patch_shape=patch_shape, instance_mask=instance_mask,
                                num_workers=num_workers, normalize='sample', sparse=sparse, padding_for_full_conv=16,
                                **kwargs))


def generate_test_dataset(datadir="/data/scannet-voxelized-sdf", num_workers=1, limit=None,
                          test_file="/data/scannet-voxelized-sdf/test.csv", grid_step=None, **kwargs):
    if grid_step is None:
        padding_for_full_conv = 8
        normalize = 'sample'
    else:
        padding_for_full_conv = None
        normalize = 'dataset'
    test_df = pd.read_csv(test_file, header=None, index_col=False)
    if limit is not None:
        assert isinstance(limit, int) and limit > 0
        test_df = test_df.iloc[:limit]
    inputs, labels = read_vox_and_label_files(datadir, test_df, num_workers=num_workers)
    return VoxelisedScanNetDataset((inputs, labels), num_workers=num_workers, grid_step=grid_step,
                                   dataset_multiplier=None, one_hot_encode_instances=False,
                                   padding_for_full_conv=padding_for_full_conv,
                                   normalize=normalize, **kwargs)