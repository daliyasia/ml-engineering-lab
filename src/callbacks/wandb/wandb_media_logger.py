import os
from typing import Any, List
from lightning import LightningModule, Trainer

import matplotlib.pyplot as plt
from lightning.pytorch.callbacks.callback import Callback
import numpy as np
import wandb
from src.models.utils.box_utils import decode, py_nms_torch
import torch.nn.functional as F
import torch
from pytorch_lightning.utilities import rank_zero_only
from pathlib import Path

class UploadCheckpointsAsArtifact(Callback):
    """Upload checkpoints to wandb as an artifact, at the end of run."""

    def __init__(self, ckpt_dir: str = "checkpoints/", upload_best_only: bool = False):
        self.ckpt_dir = ckpt_dir
        self.upload_best_only = upload_best_only

    @rank_zero_only
    def on_keyboard_interrupt(self, trainer, pl_module):
        self.on_train_end(trainer, pl_module)

    @rank_zero_only
    def on_train_end(self, trainer, pl_module):
        logger = trainer.logger
        experiment = logger.experiment

        ckpts = wandb.Artifact("experiment-ckpts", type="checkpoints")

        if self.upload_best_only:
            ckpts.add_file(trainer.checkpoint_callback.best_model_path)
        else:
            for path in Path(self.ckpt_dir).rglob("*.ckpt"):
                ckpts.add_file(str(path))

        experiment.log_artifact(ckpts)

class WatchModel(Callback):
    """Make wandb watch model at the beginning of the run."""

    def __init__(self, log: str = "gradients", log_freq: int = 100):
        self.log_ = log
        self.log_freq = log_freq

    @rank_zero_only
    def on_train_start(self, trainer, pl_module):
        logger = trainer.logger
        
        logger.watch(model=trainer.model, log=self.log_, log_freq=self.log_freq, log_graph=True)
               
class WanDbLoggerImagePredictions(Callback):
    def __init__(self,class_labels, step_pred, samples ):
        self.check_data_val = True
        self.check_data_train = True
        self.class_labels = dict(class_labels)
        self.epochs = 0
        self.step_pred = step_pred
        self.samples = samples
        pass
    
    def on_sanity_check_start(self, trainer, pl_module):
        self.ready = False

    def on_sanity_check_end(self, trainer, pl_module):
        """Start executing this callback only after all validation sanity checks end."""
        self.ready = True
        
    def on_train_epoch_end(self, trainer, pl_module):
        self.epochs += 1    
        
    def on_validation_epoch_end(self, trainer, pl_module): 
        if self.epochs % self.step_pred == 0:#self.ready:
            batch = next(iter(trainer.datamodule.val_dataloader()))
            imgs, targets = batch
            imgs, targets = imgs[:self.samples], targets[:self.samples]
            imgs = imgs.to(pl_module.device)
            outs = pl_module(imgs)
            locs = outs[0]
            confs =  F.softmax(outs[1], dim=-1)
            targets = [target.to(pl_module.device) for target in targets]
            #loss_l, loss_c = pl_module.criterion(outs, pl_module.priors, targets, pl_module.device)
            boxes_list = []
            targets_list = []
            
            for num_img in range(0,len(targets)):
                target = targets[num_img][:,0:4]
                
                loc = locs[num_img]
                conf = confs[num_img]
                
                target[:, 0::2] *= pl_module.priorbox.image_size[0]
                target[:, 1::2] *= pl_module.priorbox.image_size[1]
               
                boxes = decode(loc.data.squeeze(0), pl_module.priors.data, pl_module.criterion.variance)
                boxes[:, 0::2] *= pl_module.priorbox.image_size[0]
                boxes[:, 1::2] *= pl_module.priorbox.image_size[1]
                scores = conf[:, 1]
                inds = torch.where(scores > 0.2)[0]
                boxes = boxes[inds]
                scores = scores[inds]
                order = scores.argsort()
                boxes = boxes[order]
                scores = scores[order]
                dets = torch.hstack((boxes, scores[:, np.newaxis]))
                keep = py_nms_torch(dets,0.4)
                dets = dets[keep, :]
                                
                boxes_list.append(dets[:,0:4])
                targets_list.append(target)
            
            
            for id,img in enumerate(imgs):
                im = img.cpu().numpy().transpose(1,2,0)+[104, 117, 123]
                im = im[...,::-1]
                
                trainer.logger.experiment.log(
                {
                    f"Data Validation/epoch_{self.epochs}": [
                    wandb.Image(
                        im,
                        boxes={
                            "targets": {
                                "box_data": [
                                    {
                                        "position": {
                                            "minX": int(box[0].item()),
                                            "minY": int(box[1].item()),
                                            "maxX": int(box[2].item()),
                                            "maxY": int(box[3].item()),
                                        },
                                        "class_id": 0,
                                        "box_caption": self.class_labels[0],
                                        "domain": "pixel",
                                    }
                                    for box in targets_list[id].cpu()
                                ],
                                "class_labels":self.class_labels
                            }
                            ,
                            "predictions": {
                                "box_data": [
                                    {
                                        "position": {
                                            "minX": int(box[0].item()),
                                            "minY": int(box[1].item()),
                                            "maxX": int(box[2].item()),
                                            "maxY": int(box[3].item()),
                                        },
                                        "class_id": 1,
                                        "box_caption": self.class_labels[0][1],
                                        "domain": "pixel",
                                    }
                                    for box in boxes_list[id].cpu()
                                ],
                                "class_labels":self.class_labels
                            }
                        },
                    )
                    ]
                }
                )
                    
      
    '''def on_validation_batch_end(self,trainer, pl_module, outputs, batch, batch_idx):
        print(1)
        if batch_idx == 2:
            if self.check_data_val and self.ready:
                self.check_data_val = False
                imgs, boxes = batch
                boxes_pred = outputs['preds']
                for id,img in enumerate(imgs):
                    im = img.cpu().numpy().transpose(1,2,0)+[104, 117, 123]
                    im = im[...,::-1]
                    
                    trainer.logger.experiment.log(
                    {
                        f"Data Validation/{trainer.logger.experiment.name}": [
                        wandb.Image(
                            im,
                            boxes={
                                "targets": {
                                    "box_data": [
                                        {
                                            "position": {
                                                "minX": int(box[0].item()),
                                                "minY": int(box[1].item()),
                                                "maxX": int(box[2].item()),
                                                "maxY": int(box[3].item()),
                                            },
                                            "class_id": 0,
                                            "box_caption": self.class_labels[0],
                                            "domain": "pixel",
                                        }
                                        for box in boxes[id].cpu()
                                    ],
                                    "class_labels":self.class_labels
                                }
                                ,
                                "predictions": {
                                    "box_data": [
                                        {
                                            "position": {
                                                "minX": int(box[0].item()),
                                                "minY": int(box[1].item()),
                                                "maxX": int(box[2].item()),
                                                "maxY": int(box[3].item()),
                                            },
                                            "class_id": 1,
                                            "box_caption": self.class_labels[1],
                                            "domain": "pixel",
                                        }
                                        for box in boxes_pred[id].cpu()
                                    ],
                                    "class_labels":self.class_labels
                                }
                            },
                        )
                        ]
                    }
                    )
                    
        else:
            pass'''
        
    
'''class LogConfusionMatrix(Callback):
    """Generate confusion matrix every epoch and send it to wandb.
    Expects validation step to return predictions and targets.
    """

    def __init__(self):
        self.preds = []
        self.targets = []
        self.ready = True

    def on_sanity_check_start(self, trainer, pl_module) -> None:
        self.ready = False

    def on_sanity_check_end(self, trainer, pl_module):
        """Start executing this callback only after all validation sanity checks end."""
        self.ready = True

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        """Gather data from single batch."""
        if self.ready:
            self.preds.append(outputs["preds"])
            self.targets.append(outputs["targets"])

    def on_validation_epoch_end(self, trainer, pl_module):
        """Generate confusion matrix."""
        if self.ready:
            logger = get_wandb_logger(trainer)
            experiment = logger.experiment

            preds = torch.cat(self.preds).cpu().numpy()
            targets = torch.cat(self.targets).cpu().numpy()

            confusion_matrix = metrics.confusion_matrix(y_true=targets, y_pred=preds)

            # set figure size
            plt.figure(figsize=(14, 8))

            # set labels size
            sn.set(font_scale=1.4)

            # set font size
            sn.heatmap(confusion_matrix, annot=True, annot_kws={"size": 8}, fmt="g")

            # names should be uniqe or else charts from different experiments in wandb will overlap
            experiment.log({f"confusion_matrix/{experiment.name}": wandb.Image(plt)}, commit=False)

            # according to wandb docs this should also work but it crashes
            # experiment.log(f{"confusion_matrix/{experiment.name}": plt})

            # reset plot
            plt.clf()

            self.preds.clear()
            self.targets.clear() '''      
