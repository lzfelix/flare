from typing import List, Dict, Any, Optional

import tqdm

from flare.history import ModelHistory

OptionalLogs = Optional[Dict[str, Any]]

class Callback:
    def on_epoch_begin(self, epoch: int, ModelHistory) -> None:
        pass

    def on_epoch_end(self, epoch: int, ModelHistory) -> None:
        pass

    def on_batch_begin(self, batch: int, ModelHistory) -> None:
        pass

    def on_batch_end(self, batch: int, ModelHistory) -> None:
        pass


class CallbacksContainer:

    def __init__(self, callbacks: List[Callback]):
        self.callbacks = callbacks or list()

    def append(self, callback: Callback) -> None:
        self.callbacks.append(callback)

    def on_epoch_begin(self, epoch: int, logs: ModelHistory) -> None:
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch, logs)

    def on_epoch_end(self, epoch: int, logs: ModelHistory) -> None:
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, logs)

    def on_batch_begin(self, batch: int, logs: ModelHistory) -> None:
        for callback in self.callbacks:
            callback.on_batch_begin(batch, logs)

    def on_batch_end(self, batch: int, logs: ModelHistory) -> None:
        for callback in self.callbacks:
            callback.on_batch_end(batch, logs)


class ProgressBar(Callback):
    # https://github.com/ncullen93/torchsample/blob/master/torchsample/callbacks.py

    @staticmethod
    def _get_metrics(logs: Dict[str, List[float]]) -> float:
        return {key: value[-1]
                for key, value in logs.items()
                if 'loss' in key or 'acc' in key}

    def __init__(self, n_batches: int, n_epochs: int):
        self.n_batches = n_batches
        self.n_epochs = n_epochs

        self.last_val_loss = None
        self.last_val_acc = None
        self.progbar = None

        super().__init__()

    def on_epoch_begin(self, epoch: int, logs: ModelHistory) -> None:
        self.progbar = tqdm.tqdm(total=self.n_batches, unit=' batches')
        self.progbar.set_description(f'Epoch {epoch}/{self.n_epochs}')

    def on_epoch_end(self, epoch: int, logs: ModelHistory) -> None:
        val_metrics = self._get_metrics(logs.val_logs)
        trn_metrics = self._get_metrics(logs.trn_logs)
        trn_metrics.update(val_metrics)

        self.progbar.set_postfix(trn_metrics)
        self.progbar.update()
        self.progbar.close()

    def on_batch_begin(self, batch: int, logs: ModelHistory) -> None:
        self.progbar.update(n=1)

    def on_batch_end(self, batch: int, logs: ModelHistory) -> None:
        trn_metrics = self._get_metrics(logs.trn_logs)
        self.progbar.set_postfix(trn_metrics)
