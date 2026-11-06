import torch
from torchmetrics import Metric

from src.models.utils.metric_utils import intersection_over_union


class LprAccuracy(Metric):
    def __init__(self, threshold=0.5, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.threshold = threshold
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("length", default=torch.tensor(0), dist_reduce_fx=None)

    def update(self, preds, target, lengths):

        start = 0
        for i, length in enumerate(lengths):
            label = target[start : start + length]
            start += length
            correct_predictions = torch.isin(torch.tensor(preds[i]).cuda(), label)
            self.correct += correct_predictions.sum()
        self.total += target.numel()

    def compute(self):
        return self.correct / self.total


class LprPrecision(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("true_positives", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("false_positives", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("length", default=torch.tensor(0), dist_reduce_fx=None)

    def update(self, preds, target, lengths):
        start = 0
        for i, length in enumerate(lengths):
            label = target[start : start + length]
            start += length
            pred = torch.tensor(preds[i]).cuda()
            correct_predictions = torch.isin(pred, label)
            self.true_positives += correct_predictions.sum()
            self.false_positives += torch.count_nonzero(correct_predictions.eq(False)).item()
            self.false_positives += max(pred.numel() - label.numel(), 0)

    def compute(self):
        return self.true_positives / (self.true_positives + self.false_positives)


class LprRecall(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("true_positives", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("false_negatives", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("length", default=torch.tensor(0), dist_reduce_fx=None)

    def update(self, preds, target, lengths):
        start = 0
        for i, length in enumerate(lengths):
            label = target[start : start + length]
            start += length
            pred = torch.tensor(preds[i]).cuda()
            correct_predictions = torch.isin(pred, label)
            self.true_positives += correct_predictions.sum()
            self.false_negatives += max(label.numel() - pred.numel(), 0)

    def compute(self):
        return self.true_positives / (self.true_positives + self.false_negatives)


class LPRF1Score(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("true_positives", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("false_positives", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("false_negatives", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("length", default=torch.tensor(0), dist_reduce_fx=None)

    def update(self, preds, target, lengths):
        start = 0
        for i, length in enumerate(lengths):
            label = target[start : start + length]
            start += length
            pred = torch.tensor(preds[i]).cuda()
            correct_predictions = torch.isin(pred, label)
            self.true_positives += correct_predictions.sum()
            self.false_negatives += max(label.numel() - pred.numel(), 0)
            self.false_positives += torch.count_nonzero(correct_predictions.eq(False)).item()

    def compute(self):
        precision = self.true_positives / (self.true_positives + self.false_positives)
        recall = self.true_positives / (self.true_positives + self.false_negatives)
        f1_score = 2 * (precision * recall) / (precision + recall)
        return f1_score


class OnClassDetPrecision(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("true_positives", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("false_positives", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("iou_thr", default=torch.tensor(0), dist_reduce_fx=None)

    def update(self, preds, targets, iou_thr):

        list_correct = []

        for idx, target in enumerate(targets):
            correct_predictions = intersection_over_union(target, preds[idx], iou_thr)
            list_correct.append(correct_predictions)
            self.true_positives += sum(correct_predictions)

        self.false_positives += torch.count_nonzero(torch.cat(list_correct) == 0).item()

    def compute(self):
        return self.true_positives / (self.true_positives + self.false_positives)


class OnClassDetRecall(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("true_positives", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("false_negatives", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("iou_thr", default=torch.tensor(0), dist_reduce_fx=None)

    def update(self, preds, targets, iou_thr):

        list_correct = []

        for idx, target in enumerate(targets):
            correct_predictions = intersection_over_union(target, preds[idx], iou_thr)
            list_correct.append(correct_predictions)
            self.true_positives += sum(correct_predictions)
        self.false_negatives += max(
            sum([pred.numel() for pred in preds]) - sum([target.numel() for target in targets]), 0
        )

    def compute(self):
        return self.true_positives / (self.true_positives + self.false_negatives)


class OnClassDetF1Score(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("true_positives", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("false_positives", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("false_negatives", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("iou_thr", default=torch.tensor(0), dist_reduce_fx=None)

    def update(self, preds, targets, iou_thr):

        list_correct = []

        for idx, target in enumerate(targets):
            correct_predictions = intersection_over_union(target, preds[idx], iou_thr)
            list_correct.append(correct_predictions)
            self.true_positives += sum(correct_predictions)

        self.false_negatives += max(
            sum([pred.numel() for pred in preds]) - sum([target.numel() for target in targets]), 0
        )
        self.false_positives += torch.count_nonzero(torch.cat(list_correct).eq(False)).item()

    def compute(self):
        precision = self.true_positives / (self.true_positives + self.false_positives)
        recall = self.true_positives / (self.true_positives + self.false_negatives)
        f1_score = 2 * (precision * recall) / (precision + recall)
        return f1_score
