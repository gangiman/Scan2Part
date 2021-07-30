from torchvision.transforms import Compose
from torch.utils.data import Dataset


class ScanNetDataset(Dataset):
    def __init__(self, data, transforms=None):
        self.data = data
        self.num_samples = len(self.data)
        self.transform = None
        if transforms is not None:
            self.transform = Compose(transforms)

    def load_sample(self, item):
        raise NotImplementedError()

    def __getitem__(self, item):
        sample = self.load_sample(item)
        if self.transform is not None:
            return self.transform(sample)
        else:
            return sample

    def __len__(self):
        return self.num_samples


class VoxelisedScanNetDataset(ScanNetDataset):
    def load_sample(self, item):
        scan_vox, labels = self.data[item]
        sample = {'input': scan_vox}
        sample.update(labels)
        return sample


class SparseScanNetDataset(ScanNetDataset):
    def load_sample(self, item):
        return self.data[item]
