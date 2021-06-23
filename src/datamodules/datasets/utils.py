

# TODO: make a separate function

def compute_weights(self, num_classes, weights_mode, ignore_index=None, mapped_key='semantic', sparse=False):
    class_counts = torch.zeros(num_classes, dtype=torch.long)
    if sparse:
        for *_, labels in tqdm(
                DataLoader(self, batch_size=self.num_workers * 4,
                           num_workers=self.num_workers, collate_fn=self.collate_fn),
                leave=False, unit='batch', desc="Counting classes for weights"):
            # print('I AM HERE')
            _ind, _counts = (labels - 1).unique(return_counts=True)  # (labels - 1) from 0 to 19
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
    class_counts = class_counts[slice_from:]  # len = 20
    weights = getattr(class_counts[slice_from:], weights_mode)() / class_counts.to(torch.float)
    return weights, class_counts


def _compute_maximum_number_of_instances(self):
    _key = self.instance_mask + "_size"
    # print(self[0].keys())
    return torch.stack([self[_ind][_key] for _ind in range(self.num_samples)]).max()
