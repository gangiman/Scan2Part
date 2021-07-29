#!/usr/bin/env python
# coding: utf-8
import pickle
from itertools import product
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from tqdm import tqdm


def normalize_labels(t):
    uniqs = np.sort(np.unique(t))
    pos_uniqs = uniqs[uniqs >= 0]
    max_class = pos_uniqs.max() + 2
    remap = np.zeros(max_class, dtype=np.int)
    remap[pos_uniqs] = np.arange(0, pos_uniqs.shape[0], dtype=np.int)
    remap[-1] = 0
    return remap[t]


def prepare_color_tensor_for_plot(color):
    color_mask = (color > 0).any(axis=3)
    color_255 = (color * 255).astype(np.uint8)
    pc = np.stack(np.where(color_mask)).T
    #     output = np.zeros(color_255.shape[:3], dtype=np.int32)
    #     unique_colors, inverse = np.unique(color_255[color_mask],axis=0, return_inverse=True)
    #     output[color_mask] = inverse + 1
    col = color_255[color_mask]
    pc = pc / pc.max() - .5
    pc_col = np.sum(col[:, :3].astype(np.uint32) * np.array([1, 256, 256 ** 2])[::-1], axis=1)
    return pc, pc_col


def get_neighbours(shape, xyz, neighbour_radius):
    shape = np.array(shape)
    xyz = np.array(xyz)
    segment = tuple(range(-neighbour_radius, neighbour_radius + 1))
    cube = np.array(list(product(*[segment] * 3)))
    cube = np.r_[cube[:cube.shape[0]//2], cube[cube.shape[0]//2 + 1:]]
    offset_cube = xyz + cube
    return offset_cube[((0 <= offset_cube) & (offset_cube < shape)).all(axis=1)]


def get_mean_color(col_arr):
    # squared mean
    col_arr = col_arr.astype(np.float32)
    return np.round(np.sqrt(
        (col_arr * col_arr).sum(axis=0) / col_arr.shape[0]
    )).astype(np.uint8)


def main(
        scannet_dir="/home/ishvlad/datasets/scannet-vox/scans",
        paths="data/split/scannet.2cm.onebatch.csv",
        glob_pattern='*/*.pkl',
        num_workers=8
):
    pkls = list(Path(scannet_dir).glob(glob_pattern))
    with ProcessPoolExecutor(num_workers) as executor:
        for _new_sample, _save_path in tqdm(
                executor.map(process_sample, pkls), total=len(pkls), leave=True):
            _save_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(str(_save_path), _new_sample)


def process_sample(label_file: Path):
    new_path = Path('/home/anotchenko', *label_file.parts[3:]).with_suffix('.npy')
    print(f"processing {label_file} -> {new_path}")
    sample = pickle.load(label_file.open('rb'))
    color = sample['color'][0]
    sem_labels = sample['mask'][0]
    inst_labels = sample['object'][0]
    sdf = sample['sdf'][0]
    color_255 = (color * 255).astype(np.uint8)
    sem_mask = (sem_labels > -1)
    t_shape = sem_mask.shape
    inst_mask = inst_labels > -1
    nnz_mask = inst_mask | sem_mask
    inst_masks = [
        inst_labels == _i
        for _i in np.unique(inst_labels)
        if _i > -1
    ]
    new_color_255 = color_255.copy()
    color_mask = (new_color_255 > 0).any(axis=3)
    mask_for_coloring = nnz_mask & (~color_mask)
    print(f"{mask_for_coloring.sum()}, voxels to color")
    ignore_instance_mask = False
    neighbour_radius = 1
    while mask_for_coloring.any():
        increase_radius = True
        for _xyz in zip(*np.where(mask_for_coloring)):
            the_inst_mask = next(_im for _im in inst_masks if _im[_xyz])
            neighbour_voxels = get_neighbours(t_shape, _xyz, neighbour_radius)
            neighbours_with_color = neighbour_voxels[color_mask[tuple(neighbour_voxels.T)]]
            neigh_c_same_instance = neighbours_with_color[the_inst_mask[tuple(neighbours_with_color.T)]]
            if neigh_c_same_instance.size > 0:
                neigh_color_same_inst = new_color_255[tuple(neigh_c_same_instance.T)]
                new_color_255[_xyz] = get_mean_color(neigh_color_same_inst)
                increase_radius = False
            elif ignore_instance_mask and neighbours_with_color.size > 0:
                neigh_colors = new_color_255[tuple(neighbours_with_color.T)]
                new_color_255[_xyz] = get_mean_color(neigh_colors)
        color_mask = (new_color_255 > 0).any(axis=3)
        mask_for_coloring = nnz_mask & (~color_mask)
        print(f"{mask_for_coloring.sum()} voxels left to color")
        if increase_radius:
            neighbour_radius += 1
            print(f"increased radius to {neighbour_radius}")
            if not ignore_instance_mask and neighbour_radius > 5:
                ignore_instance_mask = True
                neighbour_radius = 1
                print("Ignoring instance masks")
    print('All voxels are colored, saving')
    new_sample = {
        'coords': np.stack(np.where(nnz_mask)).T,
        'sdf': sdf[nnz_mask],
        'mask': sem_labels[nnz_mask],
        'object': inst_labels[nnz_mask],
        'color': new_color_255[nnz_mask]
    }
    return new_sample, new_path


if __name__ == '__main__':
    main()
