import k3d
import torch
from k3d.colormaps.matplotlib_color_maps import jet
import numpy as np


def get_colormap(cm=jet, inverse=False):
    jet_cm_int = np.round(np.array(cm).reshape(-1, 4)[:, 1:]*255).astype(np.int64)
    if inverse:
        jet_cm_final = jet_cm_int.dot(np.array([16**4, 16**2, 16**0], dtype=np.int64))
    else:
        jet_cm_final = jet_cm_int.dot(np.array([16**0, 16**2, 16**4], dtype=np.int64))

    def get_colormap(n_classes):
        assert n_classes < 256
        if n_classes == 2:
            return 0x00ff00, 0xff0000
        else:
            return jet_cm_final[np.linspace(0, 255, n_classes, dtype=np.int)]
    return get_colormap


def plot_voxels(np_voxels, **kwargs):
    colormap_generator = get_colormap()
    n_classes = np.max(np_voxels)
    plot = k3d.plot(**kwargs)
    obj = k3d.voxels(np_voxels, colormap_generator(n_classes))
    plot += obj
#     plot.display()
    return plot


def plot_3d_points_as_k3d_html(coords, color):
    """_id = _i * batch_idx"""
    shape = tuple(coords.max(axis=0) + 1)
    point_cloud = coords / coords.max() - .5
    pc_col = np.sum(color[:, :3].astype(np.uint32) * np.array([1, 256, 256 ** 2])[::-1], axis=1)

    plot = k3d.plot()
    point_size = 1 / (max(shape) + 20)  # 0.005
    plot += k3d.points(point_cloud,
                       pc_col.astype(np.uint32),
                       point_size=point_size,
                       shader="flat")
    plot.display()
    # wandb.log({f"val/k3d_point_cloud_{_id}": wandb.Html(html_code)})
    return plot.get_snapshot()


def plot_3d_voxels_as_k3d_html(coords, preds):
    if isinstance(coords, torch.Tensor):
        coords = coords.cpu().detach().numpy()
    if isinstance(preds, torch.Tensor):
        preds = preds.cpu().detach().numpy()
    if len(preds.shape) > 1:
        preds = np.argmax(preds, axis=1)
    shape = tuple(coords.max(axis=0) + 1)
    voxels = np.zeros(shape, dtype=np.uint8)
    voxels[tuple(coords.T)] = preds + 1
    plot = plot_voxels(voxels)
    return plot.get_snapshot()
