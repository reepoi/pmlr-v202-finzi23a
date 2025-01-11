import shutil
from pathlib import Path
from weakref import proxy

import orbax.checkpoint
from flax.training import orbax_utils
import lightning.pytorch as pl


class LogStats(pl.callbacks.Callback):
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.log('train_loss', outputs['loss'], batch_size=len(batch), on_epoch=True, prog_bar=True)
        self.log('train_loss_ema', outputs['loss_ema'], batch_size=len(batch), on_epoch=True, prog_bar=True)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.log('val_relative_error', outputs['val_relative_error'], batch_size=len(batch), on_epoch=True, prog_bar=True)


class ModelCheckpoint(pl.callbacks.ModelCheckpoint):
    CHECKPOINT_EQUALS_CHAR = '_'

    @staticmethod
    def get_checkpoint_directories(filepath):
        filepath = Path(filepath)
        return filepath.parent/filepath.stem, filepath.parent/f'{filepath.stem}_ema'

    def _save_checkpoint(self, trainer: "pl.Trainer", filepath: str) -> None:
        params = trainer.lightning_module.params
        params_ema = trainer.lightning_module.params_ema
        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        for ckpt, directory in zip((params, params_ema), self.get_checkpoint_directories(filepath)):
            save_args = orbax_utils.save_args_from_target(ckpt)
            orbax_checkpointer.save(directory, ckpt, save_args=save_args, force=True)

        self._last_global_step_saved = trainer.global_step
        self._last_checkpoint_saved = filepath

        # notify loggers
        if trainer.is_global_zero:
            for logger in trainer.loggers:
                logger.after_save_checkpoint(proxy(self))

    def _remove_checkpoint(self, trainer: "pl.Trainer", filepath: str) -> None:
        """Calls the strategy to remove the checkpoint file."""
        for directory in self.get_checkpoint_directories(filepath):
            shutil.rmtree(directory)
