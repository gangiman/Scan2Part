import k3d
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
