import logging
from typing import List, Dict, Tuple, Any, Optional

import torch
import numpy as np
import tensorboardX
from tqdm import auto as tqdm

from flare.history import ModelHistory
from requests_futures.sessions import FuturesSession


class Callback:
    """A basic callback prototype from which all callbacks should inherit."""
    def on_epoch_begin(self, epoch: int, logs: ModelHistory) -> None:
        pass

    def on_epoch_end(self, epoch: int, logs: ModelHistory) -> None:
        pass

    def on_batch_begin(self, batch: int, logs: ModelHistory) -> None:
        pass

    def on_batch_end(self, batch: int, logs: ModelHistory) -> None:
        pass

    def on_train_end(self) -> None:
        pass

    @staticmethod
    def _log(message: str, v_threshold: int, verbose_level: int = 1, is_print: bool = False):
        if verbose_level <= v_threshold:
            fn = print if is_print else logging.info
            fn(message)


class CallbacksContainer:
    """Class that coordinates how and when all callbacks are fired during model training."""

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
        logs.flush_batch_data()

    def on_train_end(self) -> None:
        for callback in self.callbacks:
            callback.on_train_end()


class ProgressBar(Callback):
    def __init__(self, n_batches: int, n_epochs: int):
        self.n_batches = n_batches
        self.n_epochs = n_epochs

        self.last_val_loss = None
        self.last_val_acc = None
        self.progbar = None

        super().__init__()

    @staticmethod
    def _get_latest_metrics(logs: Dict[str, List[float]]) -> Dict[str, float]:
        return {key: value[-1] for key, value in logs.items()}

    def on_epoch_begin(self, epoch: int, logs: ModelHistory) -> None:
        self.progbar = tqdm.tqdm(total=self.n_batches, unit=' batches')
        self.progbar.set_description(f'Epoch {epoch}/{self.n_epochs}')

    def on_epoch_end(self, epoch: int, logs: ModelHistory) -> None:
        val_metrics = self._get_latest_metrics(logs.val_logs)
        display_metrics = self._get_latest_metrics(logs.trn_logs)
        display_metrics.update(val_metrics)

        self.progbar.set_postfix(display_metrics)
        self.progbar.close()

    def on_batch_begin(self, batch: int, logs: ModelHistory) -> None:
        self.progbar.update(n=1)

    def on_batch_end(self, batch: int, logs: ModelHistory) -> None:
        self.progbar.set_postfix(logs.batch_data, refresh=False)


class _MetricMonitorCallback(Callback):
    def get_metric(self, utility: str, metric: str, logs: ModelHistory) -> float:
        if metric not in logs.val_logs and metric not in logs.trn_logs:
            raise RuntimeError(f'Metric for {utility} ({metric}) not in the ' +
                                'model')

        current_metric = logs.val_logs.get(metric) or logs.trn_logs.get(metric)
        return current_metric[-1]


class EarlyStopping(_MetricMonitorCallback):
    """If the selected metric does not improve by delta after a given amount of epochs, stops the training.

    # Arguments
        metric_name: The name of the metric to be watched.
        patience: For how many epochs to wait before stopping the training.
        delta: The smallest variation required to reset the patience counter.
        verbosity: `0`: silent, `1`: notifies whether the metric has improved or not.
    """

    # TODO: currently this only supports decreasing metrics (ie. loss)
    def __init__(self,
                 metric_name: str,
                 patience: int,
                 delta: float,
                 verbosity: int = 0):
        self.metric_name = metric_name
        self.patience = patience
        self.ticks = 0
        self.delta = delta
        self.previous_metric = np.inf
        self.verbosity = verbosity
    
    def on_epoch_end(self, epoch: int, logs: ModelHistory) -> None:
        current_metric = self.get_metric('patience', self.metric_name, logs)
        current_delta = self.previous_metric - current_metric

        if current_delta < self.delta:
            self.ticks += 1
            self._log(f'{self.metric_name} did not improve from ' +
                      f'{self.previous_metric} in {self.ticks} epochs.',
                      self.verbosity)

            if self.ticks >= self.patience:
                logs.set_stop_training()
        else:
            self._log(f'{self.metric_name} improved from {self.previous_metric} ' +
                      f'to {current_metric}', self.verbosity)

            self.previous_metric = current_metric
            self.ticks = 0


class Checkpoint(_MetricMonitorCallback):
    """Persits a model whenever `metric[t-1] - metric[t] < delta`

    # Arguments
        metric_name: The name of the metric to be watched.
        min_delta: If the selected metric decreases by at least min_delta, the model is persisted.
        increasing: If True persists the model whenever `metric[t-1] - metric[t] > delta` instead.
        filename: Path to the model to be persisted. The extension `.pth` is automatically added.
        save_best: If False, whenever the metric differences are smaller than `min_delta` a new file
            is created. Otherwise a single file is overwritten.
        weights_only: If True does not store the model structure.
        verbosity: `0`: silent, `1`: notifies when the model is persisted.
    """
    def __init__(self,
                 metric_name: str,
                 min_delta: float,
                 filename: str,
                 save_best: bool = True,
                 increasing: bool = False,
                 weights_only: bool = False,
                 verbosity: int = 0):
        self.metric_name = metric_name
        self.min_delta = min_delta
        self.increasing = increasing

        self.save_best = save_best
        self.weights_only = weights_only
        self.filename = filename
        self.verbosity = verbosity

        self.previous_metric = 0 if increasing else np.inf

    def on_epoch_end(self, epoch: int, logs: ModelHistory) -> None:
        current_metric = self.get_metric('checkpoint', self.metric_name, logs)
        current_delta = self.previous_metric - current_metric
        if self.increasing:
            current_delta *= -1

        if current_delta >= self.min_delta:
            if self.save_best:
                destination_filename = self.filename + '.pth'
            else:
                destination_filename = self.filename + f'_{epoch}.pth'

            self._log(f'Saving model {destination_filename} ' +
                      f' - {self.metric_name} improved from ' +
                      f'{self.previous_metric} to {current_metric}',
                      self.verbosity,
                      is_print=True)
            self.previous_metric = current_metric

            if self.weights_only:
                torch.save(logs.model.state_dict(), destination_filename)
            else:
                torch.save(logs.model, destination_filename)


class TelegramNotifier(_MetricMonitorCallback):
    def __init__(self, bot_id: str, chat_id: str, metric_name: str, delta, max_workers: int = 1):
        self.bot_id = bot_id
        self.chat_id = chat_id
        self.metric_name = metric_name
        self.session = FuturesSession(max_workers=max_workers)
        self.futures = []
        self.delta = delta
        self.previous_metric = np.inf

    def on_epoch_end(self, epoch: int, logs: ModelHistory) -> None:

        def _response_hook(resp, *args, **kwargs):
            if resp.status_code != 200:
                logging.warning('Failed to deliver Telegram message with ' +
                                f'error code {resp.status_code}')

        current_metric = self.get_metric('patience', self.metric_name, logs)
        current_delta = self.previous_metric - current_metric

        if current_delta >= self.delta:
            msg = '{} improved from%20{}%20to%20{}'.format(self.metric_name,
                                                           self.previous_metric,
                                                           current_metric)
            msg_url = 'https://api.telegram.org/bot{}/sendMessage?chat_id={}&text={}'.format(
                self.bot_id, self.chat_id, msg
            )

            future = self.session.get(msg_url, hooks={
                'response': _response_hook
            })
            self.futures.append(future)
            self.previous_metric = current_metric

    def on_train_end(self) -> None:
        self.session.close()


class Tensorboard(Callback):
    """Stores model metrics at the end of each epoch."""

    def __init__(self, model_name, write_graph: Optional[Tuple[torch.nn.Module, Any]] = None):
        """
        # Arguments
            model_name: Name of the destination folder to store the models.
            write_graph: If this argument is passed, the model graph is logged in tensorboard at the end of the
                first epoch. This parameter is a tuple containing the model to be logged and a valid batch of data
                to be passed through the model.
        """
        self.writer = tensorboardX.SummaryWriter(model_name)
        self.epoch_counter = 0
        self.write_graph = write_graph

    @staticmethod
    def _get_latest_metrics(logs: Dict[str, List[float]]) -> Dict[str, float]:
        return {key: value[-1] for key, value in logs.items()}

    def on_epoch_begin(self, epoch: int, logs: ModelHistory):
        if self.epoch_counter == 0 and self.write_graph:
            self.writer.add_graph(self.write_graph[0], self.write_graph[1])
            del self.write_graph

    def on_epoch_end(self, epoch: int, logs: ModelHistory):
        self.epoch_counter += 1

        val_metrics = self._get_latest_metrics(logs.val_logs)
        display_metrics = self._get_latest_metrics(logs.trn_logs)
        display_metrics.update(val_metrics)

        for m_name, m_value in display_metrics.items():
            self.writer.add_scalar(m_name, m_value, global_step=self.epoch_counter)
