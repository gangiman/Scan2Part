import torch
import numpy as np
import random
import omegaconf
from random import choice
import MinkowskiEngine as ME
import pandas as pd
from typing import Dict
from scipy.ndimage import rotate, map_coordinates, gaussian_filter
from MinkowskiEngine.utils import sparse_collate



class ComputeInstanceMasks:
    def __init__(self, max_num_of_instances_in_sample, instance_mask):
        self.max_instances = max_num_of_instances_in_sample
        self._key = instance_mask

    def __call__(self, sample):
        masks = torch.zeros(sample[self._key].shape + (self.max_instances + 1,), dtype=torch.float32)
        masks.scatter_(3, sample[self._key].unsqueeze(-1), 1)
        sample[self._key] = masks[..., 1:]
        return sample


class NormalizeInstanceLabels:
    def __init__(self, key, label_key, background_label=0):
        self.key = key
        self.label_key = label_key # semantic_nyu20id
        self.background_label = background_label
        if background_label == 0:
            self.index_shift = 1
        else:
            self.index_shift = 0

    def __call__(self, sample):
        # classes_to_map = sample[self.key].unique() # original code
        labels = sample[self.label_key]
        classes_to_map = labels.unique()
        max_mapping_id = max(classes_to_map)
        labels_mapping = torch.zeros(max_mapping_id + 3, dtype=torch.long)
        num_instance_classes = classes_to_map[classes_to_map >= 0].shape[0]
        labels_mapping[classes_to_map[classes_to_map >= 0]] = \
            torch.arange(self.index_shift, num_instance_classes + self.index_shift,
                         dtype=torch.long)
        labels_mapping[[-2, -1]] = self.background_label
        # sample[self.key] = labels_mapping[sample[self.key]] # original code
        sample[self.key] = labels_mapping[labels] # masks
        sample[self.key + '_size'] = torch.tensor(num_instance_classes) # size
        return sample


class Normalize:
    """
    Normalizes a given input tensor to be 0-mean and 1-std.
    mean and std parameter have to be provided explicitly.
    """

    def __init__(self, mean=None, std=None, eps=1e-4, **kwargs):
        self.mean = mean
        self.std = std
        self.eps = eps

    def __call__(self, m):
        if self.mean is None or self.std is None:
            mean = m['input'].mean()
            std = m['input'].std()
        else:
            mean = self.mean
            std = self.std
        m['input'] = (m['input'] - mean) / (std + self.eps)
        return m


class ToTensor:
    def __init__(self, dtype=np.float32, **kwargs):
        self.dtype = dtype

    def __call__(self, m):
        if isinstance(m, dict):
            return {_k: torch.from_numpy(_v) for _k, _v in m.items()}
        return torch.from_numpy(m)


class MapInstancesToSemanticLabels:
    def __init__(self, labels_mapping, mapping_file='/code/mappings/norm_all_mappings_v2.csv',
                 mapped_key='semantic', normalize_labels=True, sparse=False):
        self.mapped_key = mapped_key
        self.labels_mapping = labels_mapping
        self.mapping_file = mapping_file
        label_mappings_df = pd.read_csv(mapping_file, usecols=lambda x: x == labels_mapping)
        label_mappings_df = label_mappings_df\
            .append({_col: 0 for _col in label_mappings_df}, ignore_index=True)\
            .append({_col: 0 for _col in label_mappings_df}, ignore_index=True)[labels_mapping]
        unique_labels = np.sort(label_mappings_df.unique())
        max_class = label_mappings_df.max() + 1
        if (frozenset(unique_labels) != frozenset(range(max_class))) and normalize_labels:
            remap = np.zeros(max_class, dtype=np.int)
            remap[unique_labels] = np.arange(0, unique_labels.shape[0], dtype=np.int)
            mapping = remap[label_mappings_df.values]
        else:
            mapping = label_mappings_df.values
        #######################################################################
        if not sparse:
            self.num_classes = int(np.max(mapping) + 1)
        else:
            self.num_classes = int(np.max(mapping) + 1) # for instance debug
            # self.num_classes = int(np.max(mapping))
            
        self.mapping = torch.tensor(mapping)
        #######################################################################

    def __repr__(self):
        return f"Labels Mapping for key '{self.mapped_key}'"

    def __ge__(self, other):
        """ from other to self, self <= other"""
        if int(self.labels_mapping.split('_')[-1]) >= int(other.labels_mapping.split('_')[-1]):
            return None
        cols = torch.stack([self.mapping, other.mapping])
        remap = cols.unique(dim=1)
        remap = remap[:, (remap[1] > 0) | (remap[0] == 0)]
        remap = remap[:, remap[1].argsort()]
        return remap[0]

    def __le__(self, other):
        """ from other to self, self <= other"""
        if self.num_classes > other.num_classes:
            return None
        return self.project_mappings(other)

    def project_mappings(self, other):
        df = pd.DataFrame({
            'bottom_mapping': other.mapping.numpy(),
            'up_mapping': self.mapping.numpy()
        })
        df_without_duplicates = df.drop_duplicates()
        df_without_duplicates = df_without_duplicates[(df_without_duplicates > 0).all(axis=1)]
        bottom_up_mapping = df_without_duplicates.up_mapping.copy()
        bottom_up_mapping.index = df_without_duplicates.bottom_mapping
        new_index = np.arange(0, bottom_up_mapping.index.max() + 1)
        return bottom_up_mapping.reindex(new_index, fill_value=0).values

    def __call__(self, sample):
        if isinstance(sample['parts'], np.ndarray):
            mapping = self.mapping.numpy()
        else:
            mapping = self.mapping
        sample[self.mapped_key] = mapping[sample['parts']]
        return sample


class GetInstanceMaskForLoD:
    def __init__(self, labels_mapping):
        self.labels_mapping = labels_mapping
        self.key = f"semantic_{labels_mapping}"

    def __call__(self, sample):
        labels = sample[self.label_key]
        sub_objects_mask = torch.zeros_like(labels, dtype=torch.long) # upd
        _inst_id = 1
        for _object_id in labels.unique(): # upd
            if _object_id >= 0:
                _object_mask = labels == _object_id
                _uni_lod_parts = sample[self.key][_object_mask].unique()
                for _uni_lod_part_id in _uni_lod_parts:
                    sub_objects_mask[_object_mask * (sample[self.key] == _uni_lod_part_id)] = _inst_id
                    _inst_id += 1
        sample['objects'] = sub_objects_mask
        sample['objects_size'] = torch.tensor(_inst_id - 1)
        return sample


class MapLabelsToHeads:
    def __init__(self, head_hierarchy, mapping_file):
        self.head_hierarchy = head_hierarchy
        self._mapping_file = mapping_file
        self.lod_and_set_id = {}
        self.lod_mappings = {}
        self.num_heads = 0
        for _lod, _lod_branches in self.head_hierarchy.items():
            _labels_mapping = f"map_0k_{_lod}"
            self.lod_and_set_id[_lod] = {}
            self.lod_mappings[_lod] = MapInstancesToSemanticLabels(
                _labels_mapping, mapping_file=self._mapping_file,
                mapped_key=_labels_mapping, normalize_labels=False)
            unique_classes_for_lod = set(self.lod_mappings[_lod].mapping.unique().cpu().numpy()) - {0}
            for (_parent_node_id, _children_set_ids), _num_classes in _lod_branches.items():
                ids_for_masking = unique_classes_for_lod - set(_children_set_ids)
                _branch_class_mapping = self._create_mapping_to_zero_tensor(_children_set_ids, ids_for_masking)
                self.lod_and_set_id[_lod][frozenset(_children_set_ids)] = (self.num_heads, _branch_class_mapping)
                self.num_heads += 1

    def __call__(self, sample):
        for _lod, _sids_to_mapping in self.lod_and_set_id.items():
            _labels_mapping = f"map_0k_{_lod}"
            _labels_at_level = self.lod_mappings[_lod](sample)[_labels_mapping]
            nnz_full_labels = torch.where(_labels_at_level > 0)
            unique_classes_for_lod = set(_labels_at_level.unique().cpu().numpy()) - {0}
            for _sids, (_head_id, _mapping) in _sids_to_mapping.items():
                if _sids & unique_classes_for_lod:
                    nnz_full_mapping = _mapping[_labels_at_level[nnz_full_labels]]
                    if not (nnz_full_mapping > 0).all():
                        nnz_mapping = torch.where(nnz_full_mapping > 0)
                        nnz_full_mapping = nnz_full_mapping[nnz_mapping]
                        nnz_labels = tuple(_ind[nnz_mapping] for _ind in nnz_full_labels)
                    else:
                        nnz_labels = nnz_full_labels
                    assert (nnz_full_mapping > 0).all()
                    sample[f"head_{_head_id}"] = nnz_labels, nnz_full_mapping
            del sample[_labels_mapping]
        return sample

    def num_classes(self) -> Dict[str, int]:
        return {
            f"head_{_i}": len(_nc) + 1
            for _, _sids in self.lod_and_set_id.items()
            for _nc, (_i, _) in _sids.items()
        }

    @staticmethod
    def _create_mapping_to_zero_tensor(children_set_ids, ids_for_masking):
        max_masking_id = max(ids_for_masking, default=-1)
        max_branch_set_id = max(children_set_ids)
        max_id = max(max_masking_id, max_branch_set_id)
        labels_mapping = torch.zeros(max_id + 1, dtype=torch.long)
        labels_mapping[torch.LongTensor(tuple(ids_for_masking))] = 0
        labels_mapping[torch.LongTensor(children_set_ids)] = torch.arange(
            1, len(children_set_ids) + 1, dtype=torch.long)
        return labels_mapping


class RandomFlip:
    def __init__(self, axes=(1, 2), threshold=0.5):
        self.axes = axes
        self.threshold = threshold

    def __call__(self, sample):
#         assert m.ndim in [3, 4], 'Supports only 3D (DxHxW) or 4D (CxDxHxW) images'
        for axis in self.axes:
            if np.random.uniform() > self.threshold:
                for _k, m in sample.items():
                    sample[_k] = np.flip(m, axis + int(_k == 'input'))
        return sample

class RandomRotate90:
    def __init__(self, axes=(1, 2)):
        self.axes = axes

    def __call__(self, sample):
        # pick number of rotations at random
        k = np.random.randint(0, 4)
        # rotate k times around a given plane
        for _key, m in sample.items():
            if _key == 'input':
                sample[_key] = np.rot90(m[0], k, self.axes)[np.newaxis, ...]
            else:
                sample[_key] = np.rot90(m, k, self.axes)
        return sample


class RandomRotate:
    def __init__(self, angle_spectrum=10, axes=None, order=0):
        if axes is None:
            axes = [(1, 0), (2, 1), (2, 0)]
        elif isinstance(axes, omegaconf.listconfig.ListConfig):
            axes = omegaconf.OmegaConf.to_container(axes)
        else:
            assert isinstance(axes, list) and len(axes) > 0
        self.angle_spectrum = angle_spectrum
        self.axes = axes
        self.order = order

    def __call__(self, sample):
        axis = choice(self.axes)
        angle = np.random.randint(-self.angle_spectrum, self.angle_spectrum)
        for _key, m in sample.items():
            if _key == 'input':
                mode = 'nearest'
            else:
                mode = 'constant'
            sample[_key] = rotate(m, angle, axes=axis, reshape=False, order=self.order, mode=mode, cval=-2)
        return sample

class ElasticDeformation:
    """
    Apply elasitc deformations of 3D patches on a per-voxel mesh. Assumes ZYX axis order!
    Based on: https://github.com/fcalvet/image_tools/blob/master/image_augmentation.py#L62
    """
    def __init__(self, alpha=15, sigma=3, execution_probability=0.3):
        """
        :param alpha: scaling factor for deformations
        :param sigma: smoothing factor for Gaussian filter
        """
        self.alpha = alpha
        self.sigma = sigma
        self.execution_probability = execution_probability

    def __call__(self, sample):
        if np.random.uniform() < self.execution_probability:
            m = sample['input'][0]
            assert m.ndim == 3
            dz = gaussian_filter(np.random.randn(*m.shape), self.sigma, mode="constant", cval=0) * self.alpha
            dy = gaussian_filter(np.random.randn(*m.shape), self.sigma, mode="constant", cval=0) * self.alpha
            dx = gaussian_filter(np.random.randn(*m.shape), self.sigma, mode="constant", cval=0) * self.alpha

            z_dim, y_dim, x_dim = m.shape
            z, y, x = np.meshgrid(np.arange(z_dim), np.arange(y_dim), np.arange(x_dim), indexing='ij')
            indices = z + dz, y + dy, x + dx
            for _key, m in sample.items():
                if _key == 'input':
                    sample[_key] = map_coordinates(m[0], indices, order=3, mode='reflect')[np.newaxis, ...]
                else:
                    sample[_key] = map_coordinates(m, indices, order=0, mode='reflect')
        return sample

class RandomCrop:
    def __init__(self, patch_shape=(64,64,64), label_key='parts'):
        self.patch_shape = np.asarray(patch_shape, dtype=np.int)
        self.label_key = label_key

    def __call__(self, sample):
        tensor_shape = np.asarray(sample['input'].shape[1:])
        _slice = np.floor(
            np.random.rand(3) *
            (tensor_shape - self.patch_shape + 1)
        ).astype(np.int)
        crop = tuple(slice(_slice[_i], _slice[_i] + self.patch_shape[_i]) for _i in range(3))
        _label = sample[self.label_key]
        unique_nonzero_classes_in_crop = set(np.unique(_label[crop])) - {-2, -1}
        if unique_nonzero_classes_in_crop:
            for _key, m in sample.items():
                if _key == 'input':
                    sample[_key] = m[(slice(None, None, None),) + crop]
                else:
                    sample[_key] = m[crop]
            return sample
        else:
            return self(sample)

# class RandomDropout:
#     def __init__(self, dropout_ratio=0.2):
#         """
#         https://github.com/chrischoy/SpatioTemporalSegmentation/blob/master/lib/transforms.py#L141
#         upright_axis: axis index among x,y,z, i.e. 2 for z
#         """
#         self.dropout_ratio = dropout_ratio

#     def __call__(inp):
#         coords, feats, labels = inp
#         if random.random() < self.dropout_ratio:
#             N = len(coords)
#             inds = np.random.choice(N, int(N * (1 - self.dropout_ratio)), replace=False)
#             return coords[inds], feats[inds], labels[inds]
#         return coords, feats, labels


class ToSparse:
    def __init__(self, label_key='semantic', bg_value=0, instance_mask='semantic'):
        self.label_key = label_key
        self.bg_value = bg_value
        self.instance_mask = instance_mask
    
    def __call__(self, sample: dict):
        label = sample[self.label_key]
        nnz_index = torch.where(label != self.bg_value)
        
        coordinates = torch.stack(nnz_index, dim=1).to(torch.int32)
        features = sample['input'][0][nnz_index].unsqueeze(1)
        # labels = label[nnz_index].unsqueeze(1)

        # print('\n')
        # print('coordinates:', coordinates.shape)
        # print('features:', features.shape)
        # print(sample[self.instance_mask].shape) # [39, 111, 198, 14]
        # print(sample[self.instance_mask][nnz_index].shape) # [29575, 14]
        # print(sample[self.instance_mask][nnz_index].unsqueeze(1).shape) # [29575, 1, 14]
        # print('\n')

        masks = sample[self.instance_mask][nnz_index]
        size = sample[self.instance_mask + '_size']
        instseg_dct = {'masks': masks, 'size': size, 'object_shape': features.shape[0]}

        return coordinates, features, instseg_dct