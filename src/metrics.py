import torch
import numpy as np

from argus.metrics.metric import Metric

from src import config


class MultiCategoricalAccuracy(Metric):
    name = 'multi_accuracy'
    better = 'max'

    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def reset(self):
        self.correct = 0
        self.count = 0

    def update(self, step_output: dict):
        pred = step_output['prediction']
        trg = step_output['target']
        pred = (pred > self.threshold).to(torch.float32)
        correct = torch.eq(pred, trg).all(dim=1).view(-1)
        self.correct += torch.sum(correct).item()
        self.count += correct.shape[0]

    def compute(self):
        if self.count == 0:
            raise Exception('Must be at least one example for computation')
        return self.correct / self.count


# Source: https://github.com/DCASE-REPO/dcase2019_task2_baseline/blob/master/evaluation.py
class LwlrapBase:
    """Computes label-weighted label-ranked average precision (lwlrap)."""

    def __init__(self, class_map):
        self.num_classes = 0
        self.total_num_samples = 0
        self._class_map = class_map

    def accumulate(self, batch_truth, batch_scores):
        """Accumulate a new batch of samples into the metric.
        Args:
          truth: np.array of (num_samples, num_classes) giving boolean
            ground-truth of presence of that class in that sample for this batch.
          scores: np.array of (num_samples, num_classes) giving the
            classifier-under-test's real-valued score for each class for each
            sample.
        """
        assert batch_scores.shape == batch_truth.shape
        num_samples, num_classes = batch_truth.shape
        if not self.num_classes:
            self.num_classes = num_classes
            self._per_class_cumulative_precision = np.zeros(self.num_classes)
            self._per_class_cumulative_count = np.zeros(self.num_classes,
                                                        dtype=np.int)
        assert num_classes == self.num_classes
        for truth, scores in zip(batch_truth, batch_scores):
            pos_class_indices, precision_at_hits = (
                self._one_sample_positive_class_precisions(scores, truth))
            self._per_class_cumulative_precision[pos_class_indices] += (
                precision_at_hits)
            self._per_class_cumulative_count[pos_class_indices] += 1
        self.total_num_samples += num_samples

    def _one_sample_positive_class_precisions(self, scores, truth):
        """Calculate precisions for each true class for a single sample.
        Args:
          scores: np.array of (num_classes,) giving the individual classifier scores.
          truth: np.array of (num_classes,) bools indicating which classes are true.
        Returns:
          pos_class_indices: np.array of indices of the true classes for this sample.
          pos_class_precisions: np.array of precisions corresponding to each of those
            classes.
        """
        num_classes = scores.shape[0]
        pos_class_indices = np.flatnonzero(truth > 0)
        # Only calculate precisions if there are some true classes.
        if not len(pos_class_indices):
            return pos_class_indices, np.zeros(0)
        # Retrieval list of classes for this sample.
        retrieved_classes = np.argsort(scores)[::-1]
        # class_rankings[top_scoring_class_index] == 0 etc.
        class_rankings = np.zeros(num_classes, dtype=np.int)
        class_rankings[retrieved_classes] = range(num_classes)
        # Which of these is a true label?
        retrieved_class_true = np.zeros(num_classes, dtype=np.bool)
        retrieved_class_true[class_rankings[pos_class_indices]] = True
        # Num hits for every truncated retrieval list.
        retrieved_cumulative_hits = np.cumsum(retrieved_class_true)
        # Precision of retrieval list truncated at each hit, in order of pos_labels.
        precision_at_hits = (
                retrieved_cumulative_hits[class_rankings[pos_class_indices]] /
                (1 + class_rankings[pos_class_indices].astype(np.float)))
        return pos_class_indices, precision_at_hits

    def per_class_lwlrap(self):
        """Return a vector of the per-class lwlraps for the accumulated samples."""
        return (self._per_class_cumulative_precision /
                np.maximum(1, self._per_class_cumulative_count))

    def per_class_weight(self):
        """Return a normalized weight vector for the contributions of each class."""
        return (self._per_class_cumulative_count /
                float(np.sum(self._per_class_cumulative_count)))

    def overall_lwlrap(self):
        """Return the scalar overall lwlrap for cumulated samples."""
        return np.sum(self.per_class_lwlrap() * self.per_class_weight())

    def __str__(self):
        per_class_lwlrap = self.per_class_lwlrap()
        # List classes in descending order of lwlrap.
        s = (['Lwlrap(%s) = %.6f' % (name, lwlrap) for (lwlrap, name) in
              sorted([(per_class_lwlrap[i], self._class_map[i]) for i in range(self.num_classes)],
                     reverse=True)])
        s.append('Overall lwlrap = %.6f' % (self.overall_lwlrap()))
        return '\n'.join(s)


class Lwlrap(Metric):
    name = 'lwlrap'
    better = 'max'

    def __init__(self, classes=None):
        self.classes = classes
        if self.classes is None:
            self.classes = config.classes

        self.lwlrap = LwlrapBase(self.classes)

    def reset(self):
        self.lwlrap.num_classes = 0
        self.lwlrap.total_num_samples = 0

    def update(self, step_output: dict):
        pred = step_output['prediction'].cpu().numpy()
        trg = step_output['target'].cpu().numpy()
        self.lwlrap.accumulate(trg, pred)

    def compute(self):
        return self.lwlrap.overall_lwlrap()
