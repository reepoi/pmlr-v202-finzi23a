import lightning.pytorch as pl


class LogStats(pl.callbacks.Callback):
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.log('train_loss', outputs['loss'], batch_size=len(batch), on_epoch=True, prog_bar=True)
        self.log('train_loss_ema', outputs['loss_ema'], batch_size=len(batch), on_epoch=True, prog_bar=True)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.log('val_relative_error', outputs['val_relative_error'], batch_size=len(batch), on_epoch=True, prog_bar=True)


class ModelCheckpoint(pl.callbacks.ModelCheckpoint):
    CHECKPOINT_EQUALS_CHAR = '_'
