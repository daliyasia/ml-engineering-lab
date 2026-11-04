import inspect
import os

import mlflow.pytorch
from lightning.pytorch.callbacks.callback import Callback


class MLFlowCheckpointsCallback(Callback):
    def __init__(self):
        pass
    
    def on_test_end(self, trainer, pl_module):
        results_dir = os.path.join(trainer.default_root_dir, "results")
        #data_dir = trainer.datamodule.hparams.data_dir
        loggers = trainer.loggers
        if loggers is not None:
            if len(loggers) > 0:
                try:                   
                    ckpt_path = trainer.checkpoint_callback.best_model_path
                    model_checkpoint_name = os.path.basename(ckpt_path)
                    model_checkpoint_path = os.path.dirname(ckpt_path)
                    model_best_checkpoint_path = os.path.join(model_checkpoint_path, model_checkpoint_name)
                    model_last_checkpoint_path = os.path.join(model_checkpoint_path, 'last.ckpt')
                    if os.path.exists(model_best_checkpoint_path):
                        trainer.logger.experiment.log_artifact(trainer.logger.run_id, model_best_checkpoint_path, "checkpoints")
                        trainer.logger.experiment.log_artifact(trainer.logger.run_id, model_last_checkpoint_path, "checkpoints")
                    trainer.logger.experiment.log_artifact(trainer.logger.run_id, ckpt_path, "checkpoints")
                    trainer.logger.experiment.log_artifact(trainer.logger.run_id, "./src/", "code")
                    #trainer.logger.experiment.log_artifact(trainer.logger.run_id, data_dir, "data")
                    trainer.logger.experiment.log_artifact(trainer.logger.run_id, results_dir, "results")
                    mlflow.set_tracking_uri(trainer.logger.experiment.tracking_uri)
                    with mlflow.start_run(run_id=trainer.logger.run_id):
                        mlflow.pytorch.log_model(
                            pytorch_model=pl_module, 
                            artifact_path=trainer.datamodule.hparams.name,
                            code_paths=[
                                inspect.getfile(trainer.datamodule.__class__),
                                inspect.getfile(pl_module.__class__)
                            ],
                            registered_model_name=trainer.datamodule.hparams.name,
                        )
                except:
                    pass
