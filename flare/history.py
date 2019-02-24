from collections import defaultdict

class ModelHistory:
    def __init__(self):
        self.n_epochs = 0
        self.trn_logs = defaultdict(list)
        self.val_logs = defaultdict(list)
    
    def close(self, n_epochs: int) -> None:
        self.n_epochs = n_epochs
    
    def append_trn_logs(self, key: str, value: float) -> None:
        self.trn_logs[key].append(value)
    
    def append_dev_logs(self, key: str, value: float) -> None:
        self.val_logs[key].append(value)
