import glob
import os

import torch
import wandb
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.loggers import LoggerCollection, WandbLogger


def get_wandb_logger(trainer: Trainer) -> WandbLogger:
    """Safely get Weights&Biases logger from Trainer."""

    if isinstance(trainer.logger, WandbLogger):
        return trainer.logger

    if isinstance(trainer.logger, LoggerCollection):
        for logger in trainer.logger:
            if isinstance(logger, WandbLogger):
                return logger

    raise Exception(
        "You are using wandb related callback, but WandbLogger was not found for some reason..."
    )


class WatchModel(Callback):
    """Make wandb watch model at the beginning of the run."""

    def __init__(self, log: str = "gradients", log_freq: int = 100):
        self.log = log
        self.log_freq = log_freq

    def on_train_start(self, trainer, pl_module):
        logger = get_wandb_logger(trainer=trainer)
        logger.watch(model=trainer.model, log=self.log, log_freq=self.log_freq)


class UploadCodeAsArtifact(Callback):
    """Upload all *.py files to wandb as an artifact, at the beginning of the run."""

    def __init__(self, code_dir: str):
        self.code_dir = code_dir

    def on_train_start(self, trainer, pl_module):
        logger = get_wandb_logger(trainer=trainer)
        experiment = logger.experiment

        code = wandb.Artifact("project-source", type="code")
        for path in glob.glob(os.path.join(self.code_dir, "**/*.py"), recursive=True):
            code.add_file(path)

        experiment.use_artifact(code)


class UploadCheckpointsAsArtifact(Callback):
    """Upload checkpoints to wandb as an artifact, at the end of run."""

    def __init__(self, ckpt_dir: str = "checkpoints/", upload_best_only: bool = False):
        self.ckpt_dir = ckpt_dir
        self.upload_best_only = upload_best_only

    def on_train_end(self, trainer, pl_module):
        logger = get_wandb_logger(trainer=trainer)
        experiment = logger.experiment

        ckpts = wandb.Artifact("experiment-ckpts", type="checkpoints")

        if self.upload_best_only:
            ckpts.add_file(trainer.checkpoint_callback.best_model_path)
        else:
            for path in glob.glob(os.path.join(self.ckpt_dir, "**/*.ckpt"), recursive=True):
                ckpts.add_file(path)

        experiment.use_artifact(ckpts)


class LogImagePredictions(Callback):
    """Logs a validation batch and their predictions to wandb.
    Example adapted from:
        https://wandb.ai/wandb/wandb-lightning/reports/Image-Classification-using-PyTorch-Lightning--VmlldzoyODk1NzY
    """

    def __init__(self, num_samples: int = 8):
        super().__init__()
        self.num_samples = num_samples
        self.ready = True

    def on_sanity_check_start(self, trainer, pl_module):
        self.ready = False

    def on_sanity_check_end(self, trainer, pl_module):
        """Start executing this callback only after all validation sanity checks end."""
        self.ready = True

    def on_validation_epoch_end(self, trainer, pl_module):
        if self.ready:
            logger = get_wandb_logger(trainer=trainer)
            experiment = logger.experiment

            # get a validation batch from the validation dat loader
            val_samples = next(iter(trainer.datamodule.val_dataloader()))
            val_imgs, val_labels = val_samples

            # run the batch through the network
            val_imgs = val_imgs.to(device=pl_module.device)
            logits = pl_module(val_imgs)
            preds = torch.argmax(logits, axis=-1)

            # log the images as wandb Image
            experiment.log(
                {
                    f"Images/{experiment.name}": [
                        wandb.Image(x, caption=f"Pred:{pred}, Label:{y}")
                        for x, pred, y in zip(
                            val_imgs[: self.num_samples],
                            preds[: self.num_samples],
                            val_labels[: self.num_samples],
                        )
                    ]
                }
            )
