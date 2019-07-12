from typing import Dict
import numpy as np
import torch
from sklearn import datasets
from torch import nn, optim

from flare import trainer
from flare.callbacks import EarlyStopping, Checkpoint, TelegramNotifier

BOT_SECRET_TOKEN = '<TOKEN_HERE>'
CHAT_SECRET_TOKEN = '<TOKEN_HERE>'


class SimpleNN(nn.Module):
    def __init__(self, in_features, hidden_size, n_classes):
        super().__init__()
        self.linear1 = nn.Linear(in_features, hidden_size)
        self.linear2 = nn.Linear(hidden_size, n_classes)

    def forward(self, inputs):
        x = torch.tanh(self.linear1(inputs))
        logits = torch.tanh(self.linear2(x))

        return logits

    def metric(self, prediction: torch.Tensor, ground: torch.Tensor) -> Dict[str, float]:
        amount_correct = torch.sum(prediction.argmax(-1) == ground).item()
        return {'accuracy': amount_correct}


if __name__ == '__main__':
    X, y = datasets.make_moons(n_samples=10000, noise=0.2, random_state=99)
    
    X = torch.from_numpy(np.float32(X))
    y = torch.from_numpy(y)
    
    model = SimpleNN(in_features=2, hidden_size=20, n_classes=2)
    optimizer = optim.SGD(model.parameters(), lr=1e-1)
    loss_fn = nn.CrossEntropyLoss()

    patience = EarlyStopping(3, 0.001, 'val_loss', verbosity=1)
    checkpoint = Checkpoint('val_loss', 0.001, './model', verbosity=1)
    telegram   = TelegramNotifier(bot_id=BOT_SECRET_TOKEN, chat_id=CHAT_SECRET_TOKEN, delta=1e-3, metric_name='val_loss')

    logs = trainer.train(model, X, y,
                         loss_fn, optimizer,
                         n_epochs=25,
                         batch_size=32,
                         validation_frac=0.01,
                         callbacks=[patience, checkpoint, telegram])
