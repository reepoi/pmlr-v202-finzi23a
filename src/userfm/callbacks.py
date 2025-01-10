

class Callback:
    def __init__(self):
        pass

    def before_training_step(self, i, batch):
        pass

    def after_training_step(self, i, batch, outputs):
        pass

    def before_training_epoch(self, epoch):
        pass

    def after_training_epoch(self, epoch):
        pass


class LogLoss(Callback):
    def __init__(self, writer):
        super().__init__()
        self.writer = writer

    def after_training_step(self, i, batch, outputs):
        self.loss = outputs['loss']

    def after_training_epoch(self, epoch):
        if epoch + 1 % 25 == 0:
            self.writer.write_scalars(epoch, dict(loss=self.loss))

