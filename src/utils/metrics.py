import numpy as np
import pandas as pd
from scipy import stats as stats


class EvaluateInstanceSegmentationPR:
    def __init__(self, num_classes, ap_threshold=0.50):
        # Init test set wide vars
        self.ap_threshold = ap_threshold
        # total number of classes without background-class
        self.num_classes = num_classes
        self.iou = self.inst_tuples = self.total = self.fps = self.tps = None
        self.reset()

    def reset(self):
        self.inst_tuples = []
        self.total = np.zeros(self.num_classes, dtype=np.int)
        self.fps = [[] for i in range(self.num_classes)]
        self.tps = [[] for i in range(self.num_classes)]
        self.iou = [[] for i in range(self.num_classes)]

    def _distribute_masks_to_classes(self, _instances, _semantic, sizes=None, threshold=0.25):
        _inst_ids = np.unique(_instances)
        num_instances = _inst_ids.shape[0]
        outputs = [[] for i in range(self.num_classes)]
        for gid in _inst_ids:
            # one of the instance masks
            indices = (_instances == gid)
            # predicted semantic class for that instance
            majority_cls = int(stats.mode(_semantic[indices])[0]) - 1
            if sizes is None or (sizes is not None and np.sum(indices) > threshold * sizes[majority_cls]):
                outputs[majority_cls] += [indices]
        return outputs, num_instances

    def update_rates(self, pred_instances, semantic_prediction, truth_inst_masks):
        sizes = dict(zip(*np.unique(semantic_prediction - 1, return_counts=True)))
        proposals, num_proposals = self._distribute_masks_to_classes(
            pred_instances, semantic_prediction, sizes=sizes)
        instances, num_instances = self._distribute_masks_to_classes(
            truth_inst_masks, semantic_prediction)
        self.inst_tuples.append((num_instances, num_proposals))
        for i, (_instances, _proposals) in enumerate(zip(instances, proposals)):
            # for each class
            self.total[i] += len(_instances)
            _num_proposals = len(_proposals)
            # total number of masks per class
            tp = np.zeros(_num_proposals)
            fp = np.zeros(_num_proposals)
            for pid, u in enumerate(_proposals):
                # for each proposal u of class i enumerated by pid
                is_true_positive = False
                for iid, v in enumerate(_instances):
                    iou = (u & v).sum() / (u | v).sum()
                    if iou >= self.ap_threshold:
                        self.iou[i].append(iou)
                        is_true_positive = True
                        break
                if is_true_positive:
                    tp[pid] = 1
                else:
                    fp[pid] = 1
            self.tps[i] += [tp]
            self.fps[i] += [fp]

    def compute_metrics(self):
        p = np.zeros(self.num_classes)
        r = np.zeros(self.num_classes)
        iou = np.zeros(self.num_classes)
        for i in range(self.num_classes):
            tp = np.concatenate(self.tps[i], axis=0)
            fp = np.concatenate(self.fps[i], axis=0)
            iou[i] = np.mean(self.iou[i])
            tp = np.sum(tp)
            fp = np.sum(fp)
            p[i] = tp / (tp + fp)
            r[i] = tp / self.total[i]
        pr_iou = pd.DataFrame({"precision": p, "recall": r, "iou": iou, "num_instances": self.total})
        return pr_iou, np.nanmean(p), np.nanmean(iou)