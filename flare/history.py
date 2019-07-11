from collections import defaultdict
from typing import Dict

class ModelHistory:
    def __init__(self, model):
        self.n_epochs = 0
        self.trn_logs = defaultdict(list)
        self.val_logs = defaultdict(list)
        self._stop_training = False
        self.model = model

        self.batch_data = defaultdict(list)
    
    def close(self, n_epochs: int) -> None:
        self.n_epochs = n_epochs

    def flush_batch_data(self):
        self.batch_data = defaultdict()

    def append_batch_data(self, batch_metrics: dict) -> None:
        self.batch_data.update(batch_metrics)

    def append_trn_logs(self, trn_metrics: Dict[str, float]) -> None:
        for metric, value in trn_metrics.items():
            self.trn_logs[metric].append(value)

    def append_dev_logs(self, dev_metrics: Dict[str, float]) -> None:
        for metric, value in dev_metrics.items():
            self.val_logs[metric].append(value)

    def set_stop_training(self) -> None:
        self._stop_training = True

    def should_stop_training(self) -> bool:
        return self._stop_training
