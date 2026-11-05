import os
from typing import Any, List

import torch
from lightning import LightningModule
from torchmetrics import ConfusionMatrix, F1Score, MaxMetric, MeanMetric, MinMetric
from torchmetrics.classification.accuracy import Accuracy
from torchmetrics.classification.precision_recall import Precision, Recall


class MNISTLitModule(LightningModule):
    """Example of LightningModule for MNIST classification.

    A LightningModule organizes your PyTorch code into 6 sections:
        - Initialization (__init__)
        - Train Loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Prediction Loop (predict_step)
        - Optimizers and LR Schedulers (configure_optimizers)

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["net"])

        self.net = net
        self.criterion = torch.nn.CrossEntropyLoss()

        self.train_accuracy = Accuracy(task="multiclass", num_classes=10)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=10)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=10)

        self.train_precision = Precision(task="multiclass", num_classes=10)
        self.val_precision = Precision(task="multiclass", num_classes=10)
        self.test_precision = Precision(task="multiclass", num_classes=10)

        self.train_recall = Recall(task="multiclass", num_classes=10)
        self.val_recall = Recall(task="multiclass", num_classes=10)
        self.test_recall = Recall(task="multiclass", num_classes=10)

        self.train_f1 = F1Score(task="multiclass", num_classes=10)
        self.val_f1 = F1Score(task="multiclass", num_classes=10)
        self.test_f1 = F1Score(task="multiclass", num_classes=10)

        self.train_confusion = ConfusionMatrix(task="multiclass", num_classes=10)
        self.val_confusion = ConfusionMatrix(task="multiclass", num_classes=10)
        self.test_confusion = ConfusionMatrix(task="multiclass", num_classes=10)

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        self.val_accuracy_best = MaxMetric()
        self.val_precision_best = MaxMetric()
        self.val_recall_best = MaxMetric()
        self.val_f1_best = MaxMetric()
        self.val_loss_best = MinMetric()

    def forward(self, x: torch.Tensor):
        return self.net(x)

    def on_train_start(self):
        self.val_accuracy_best.reset()
        self.val_precision_best.reset()
        self.val_recall_best.reset()
        self.val_f1_best.reset()
        self.val_loss_best.reset()

        self.train_loss.reset()
        self.val_loss.reset()
        self.test_loss.reset()

    def model_step(self, batch: Any):
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.model_step(batch)

        self.train_loss(loss)
        self.train_accuracy(preds, targets)
        self.train_precision(preds, targets)
        self.train_recall(preds, targets)
        self.train_f1(preds, targets)
        self.train_confusion(preds, targets)

        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/accuracy", self.train_accuracy, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/precision", self.train_precision, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/recall", self.train_recall, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/f1", self.train_f1, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.model_step(batch)

        self.val_loss(loss)
        self.val_accuracy(preds, targets)
        self.val_precision(preds, targets)
        self.val_recall(preds, targets)
        self.val_f1(preds, targets)
        self.val_confusion(preds, targets)

        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/accuracy", self.val_accuracy, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/precision", self.val_precision, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/recall", self.val_recall, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/f1", self.val_f1, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss, "preds": preds, "targets": targets}

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.model_step(batch)
        self.test_loss(loss)
        self.test_accuracy(preds, targets)
        self.test_precision(preds, targets)
        self.test_recall(preds, targets)
        self.test_f1(preds, targets)
        self.test_confusion(preds, targets)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/accuracy", self.test_accuracy, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/precision", self.test_precision, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/recall", self.test_recall, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/f1", self.test_f1, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss, "preds": preds, "targets": targets}

    def on_train_epoch_end(self):
        # Preserve the original metric names used by checkpoint consumers and tests.
        self.log("train/acc", self.train_accuracy.compute())

    def on_validation_epoch_end(self):
        self.log("val/acc", self.val_accuracy.compute())
        self.val_accuracy_best(self.val_accuracy.compute())  # update best so far val acc
        self.val_precision_best(self.val_precision.compute())
        self.val_recall_best(self.val_recall.compute())
        self.val_f1_best(self.val_f1.compute())
        self.val_loss_best(self.val_loss.compute())
        self.log("val_best/accuracy", self.val_accuracy_best.compute(), prog_bar=True)
        self.log("val_best/precision", self.val_precision_best.compute(), prog_bar=True)
        self.log("val_best/recall", self.val_recall_best.compute(), prog_bar=True)
        self.log("val_best/f1", self.val_f1_best.compute(), prog_bar=True)
        self.log("val_best/loss", self.val_loss_best.compute(), prog_bar=True)

    def on_test_epoch_end(self):
        self.log("test/acc", self.test_accuracy.compute())

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    _ = MNISTLitModule(None, None, None)
