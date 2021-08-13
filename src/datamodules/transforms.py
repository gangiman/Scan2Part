import torch
import numpy as np
import omegaconf
from random import choice
import pandas as pd
from typing import Dict, Sequence, Tuple, Optional
from scipy.ndimage import rotate, map_coordinates, gaussian_filter
# from src.utils.hierarchy import get_output_hierarchy


class ComputeInstanceMasks:
    def __init__(self, instance_mask='objects'):
        self.instance_mask = instance_mask

    def __call__(self, sample):
        _, _, label_dict = sample
        instance_labels = label_dict[self.instance_mask]
        num_instances = label_dict[f"{self.instance_mask}_size"]
        masks = torch.zeros((instance_labels.shape[0], num_instances), dtype=torch.float32)
        masks.scatter_(1, instance_labels.unsqueeze(-1), 1)
        label_dict[self.instance_mask] = masks
        return sample


class NormalizeInstanceLabels:
    def __init__(self, instance_label_key='objects'):
        self.instance_label_key = instance_label_key

    def __call__(self, sample):
        instance_labels = sample[self.instance_label_key]
        classes_to_map = instance_labels.unique()
        max_mapping_id = max(classes_to_map)
        labels_mapping = torch.zeros(max_mapping_id + 3, dtype=torch.long)
        num_instance_classes = classes_to_map[classes_to_map >= 0].shape[0]
        labels_mapping[classes_to_map[classes_to_map >= 0]] = \
            torch.arange(0, num_instance_classes, dtype=torch.long)
        labels_mapping[[-2, -1]] = -2
        sample[self.instance_label_key] = labels_mapping[instance_labels]
        # assumes background instance labels are < 0
        num_instance_classes = classes_to_map[classes_to_map >= 0].shape[0]
        sample[self.instance_label_key + '_size'] = torch.tensor(num_instance_classes)
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
    def __init__(
            self,
            labels_mapping: str = None,
            mapping_file: str = None,
            mapped_key: str = 'semantic',
            labels_key: str = 'parts',
            normalize_labels: bool = True,
            map_bg_to_value: int = 0
    ):
        self.mapped_key = mapped_key
        self.labels_key = labels_key
        self.labels_mapping = labels_mapping
        self.mapping_file = mapping_file
        label_mappings_df = pd.read_csv(mapping_file, usecols=lambda x: x == labels_mapping)
        label_mappings_df = label_mappings_df\
            .append({_col: 0 for _col in label_mappings_df}, ignore_index=True)\
            .append({_col: 0 for _col in label_mappings_df}, ignore_index=True)[labels_mapping]
        unique_labels = np.sort(label_mappings_df.unique())
        max_class = label_mappings_df.max() + 1
        if normalize_labels:
            remap = np.zeros(max_class, dtype=np.int)
            remap[unique_labels] = np.arange(0, unique_labels.shape[0], dtype=np.int)
            remap += map_bg_to_value
            mapping = remap[label_mappings_df.values]
        else:
            mapping = label_mappings_df.values
        self.num_classes = int(np.max(mapping) + 1)
        self.mapping = torch.tensor(mapping)

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
        if isinstance(sample[self.labels_key], np.ndarray):
            mapping = self.mapping.numpy()
        else:
            mapping = self.mapping
        sample[self.mapped_key] = mapping[sample[self.labels_key]]
        return sample


class GetInstanceMaskForLoD:
    def __init__(self, semantic_key='semantic', instance_key='object'):
        self.instance_key = instance_key
        self.semantic_key = semantic_key

    def __call__(self, sample):
        instance_labels = sample[self.instance_key]
        sub_objects_mask = torch.zeros_like(instance_labels, dtype=torch.long)
        semantic_labels = sample[self.semantic_key]
        _inst_id = 0
        for _object_id in instance_labels.unique():
            _object_mask = instance_labels == _object_id
            _uni_lod_parts = semantic_labels[_object_mask].unique()
            for _uni_lod_part_id in _uni_lod_parts:
                sub_objects_mask[_object_mask * (semantic_labels == _uni_lod_part_id)] = _inst_id
                _inst_id += 1
        sample[self.instance_key] = sub_objects_mask
        sample[f'{self.instance_key}_size'] = torch.tensor(_inst_id)
        return sample


class MapLabelsToHeads:
    def __init__(self,
                 hierarchy_file: str = None,
                 mapping_file: str = None,
                 nodes: Optional[Sequence] = None,
                 lods: Optional[Sequence] = None
                 ):
        self.head_hierarchy = get_output_hierarchy(
            hierarchy_file,
            selected_nodes=nodes,
            selected_lods=lods)
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
    def __init__(
            self,
            keys_to_sparsify: Sequence = ('semantic', 'objects', 'objects_size'),
            nnz_key='objects',
            bg_value=0,
    ):
        self.keys_to_sparsify = keys_to_sparsify
        self.bg_value = bg_value
        self.nnz_key = nnz_key
    
    def __call__(self, sample: Dict[str, torch.Tensor]):
        nnz_labels = sample[self.nnz_key]
        nnz_index = torch.where(nnz_labels >= self.bg_value)
        coordinates = torch.stack(nnz_index, dim=1).to(torch.int32)
        features = sample['input'][0][nnz_index].unsqueeze(1)
        output = {}
        for _key in self.keys_to_sparsify:
            labels = sample[_key]
            if labels.ndim > 1:
                output[_key] = labels[nnz_index]
            else:
                output[_key] = labels
        return coordinates, features, output


class PrepareSparseFeatures:
    def __init__(
            self,
            keys_to_sparsify: Tuple[str] = ('semantic', 'object'),
            nnz_key: str = 'semantic',
            bg_value: int = 0,
            add_color: bool = False
    ):
        self.keys_to_sparsify = keys_to_sparsify
        self.bg_value = bg_value
        self.nnz_key = nnz_key
        self.add_color = add_color

    def __call__(self, sample: Dict[str, torch.Tensor]):
        nnz_index = sample[self.nnz_key] >= self.bg_value
        coordinates = sample['coords'][nnz_index]
        features = sample['sdf'][nnz_index].unsqueeze(1)
        features *= 5.0
        if self.add_color:
            color = sample['color'][nnz_index, :3].to(torch.float32)
            color -= 127.5
            color /= 127.5
            features = torch.cat([features, color], dim=1)

        output = {}
        for _key in self.keys_to_sparsify:
            labels = sample[_key]
            if labels.ndim and labels.shape[0] > 1:
                output[_key] = labels[nnz_index]
            else:
                output[_key] = labels

        return coordinates, features, output
