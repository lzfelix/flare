import logging
from typing import List, Dict, Any, Optional

import tqdm
import torch
import numpy as np

from flare.history import ModelHistory
from requests_futures.sessions import FuturesSession


class Callback:
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
    def _log(message: str, v_threshold: int, verbose_level: int = 1):
        if verbose_level <= v_threshold:
            logging.info(message)


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
    def _get_latest_metrics(logs: Dict[str, List[float]]) -> float:
        return {key: value[-1] for key, value in logs.items()}

    def on_epoch_begin(self, epoch: int, logs: ModelHistory) -> None:
        self.progbar = tqdm.tqdm(total=self.n_batches, unit=' batches')
        self.progbar.set_description(f'Epoch {epoch}/{self.n_epochs}')

    def on_epoch_end(self, epoch: int, logs: ModelHistory) -> None:
        val_metrics = self._get_latest_metrics(logs.val_logs)
        display_metrics = self._get_latest_metrics(logs.trn_logs)
        display_metrics.update(val_metrics)

        self.progbar.set_postfix(display_metrics)
        self.progbar.update()
        self.progbar.close()

    def on_batch_begin(self, batch: int, logs: ModelHistory) -> None:
        self.progbar.update(n=1)

    def on_batch_end(self, batch: int, logs: ModelHistory) -> None:
        self.progbar.set_postfix(logs.batch_data, refresh=False)


class MetricMonitorCallback(Callback):
    def get_metric(self, utility: str, metric: str, logs: ModelHistory) -> float:
        if metric not in logs.val_logs and metric not in logs.trn_logs:
            raise RuntimeError(f'Metric for {utility} ({metric}) not in the ' +
                                'model')

        current_metric = logs.val_logs.get(metric) or logs.trn_logs.get(metric)
        return current_metric[-1]


class EarlyStopping(MetricMonitorCallback):
    def __init__(self,
                 patience: int,
                 delta: float,
                 metric_name: str,
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


class Checkpoint(MetricMonitorCallback):

    def __init__(self,
                 metric_name: str,
                 min_delta: float,
                 filename: str,
                 save_best: bool = True,
                 weights_only: bool = False,
                 verbosity: int = 0):
        self.metric_name = metric_name
        self.min_delta = min_delta

        self.save_best = save_best
        self.weights_only = weights_only
        self.filename = filename
        self.verbosity = verbosity

        self.previous_metric = np.inf

    def on_epoch_end(self, epoch: int, logs: ModelHistory) -> None:
        current_metric = self.get_metric('patience', self.metric_name, logs)
        current_delta = self.previous_metric - current_metric

        if current_delta >= self.min_delta:
            if self.save_best:
                destination_filename = self.filename + '.pth'
            else:
                destination_filename = self.filename + f'_{epoch}.pth'

            self.previous_metric = current_metric
            self._log(f'Saving model {destination_filename} ' +
                      f' - {self.metric_name} improved from ' +
                      f'{self.previous_metric} to {current_metric}',
                      self.verbosity)

            if self.weights_only:
                torch.save(logs.model.state_dict(), destination_filename)
            else:
                torch.save(logs.model, destination_filename)


class TelegramNotifier(MetricMonitorCallback):

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
