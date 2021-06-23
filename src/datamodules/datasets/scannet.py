from torchvision.transforms import Compose
from torch.utils.data import Dataset


class VoxelisedScanNetDataset(Dataset):
    def __init__(self, inputs, labels, transforms=None):
        self.scan_vox = inputs
        self.labels = labels
        self.num_samples = len(self.labels)
        self.transform = None
        if transforms is not None:
            self.transform = Compose(transforms)

    def __getitem__(self, item):
        sample = {'input': self.scan_vox[item]}
        sample.update(self.labels[item])
        if self.transform is not None:
            return self.transform(sample)
        else:
            return sample

    def __len__(self):
        return self.num_samples
