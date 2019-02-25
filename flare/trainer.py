from typing import List, Optional, Any

import torch
import numpy as np

from torch.optim import Optimizer
from torch import nn
from sklearn import metrics

from flare.callbacks import Callback, CallbacksContainer, ProgressBar
from flare.history import ModelHistory

_ceil = lambda x: int(np.ceil(x))

def train(model: nn.Module,
          X: torch.Tensor,
          y: torch.Tensor,
          loss_fn: Any,
          optimizer: Optimizer,
          n_epochs: int,
          batch_size: int,
          validation_frac: float,
          log_every: int = 1,
          callbacks: Optional[List[Callback]] = None) -> ModelHistory:
    """Helper function to train the model.

    # Arguments
        model: The PyTorch model
        X: Tensors representing the sample features
        y: Tensors with categorical labels
        n_epochs
        batch_size
        validation_frac: Percentage of the samples used for validation only
            after shuffling
        log_every: Evaluate the model and log its metrics after how many
            epochs
    
    # Return
        A ModelHistory object with the logs of model metrics after each epoch.
    """

    if batch_size < 1:
        raise RuntimeError('Each batch should have at least one sample.')
    callbacks_container = CallbacksContainer(callbacks or [])

    # first dimension in the amount of samples
    n_samples = X.shape[0]
    val_samples = _ceil(n_samples * validation_frac)
    trn_samples = n_samples - val_samples

    n_batches = _ceil(trn_samples / batch_size)

    # first permute before separating the validation samples
    permutations = torch.randperm(n_samples)
    X = X[permutations]
    y = y[permutations]
    
    # getting the validation samples
    X_val = X[:val_samples]
    y_val = y[:val_samples]
    
    # the remaining are train samples
    X_trn = X[val_samples:]
    y_trn = y[val_samples:]

    ticker = ProgressBar(n_batches, n_epochs)
    callbacks_container.append(ticker)

    model_history = ModelHistory()
    for epoch in range(1, n_epochs + 1):
        callbacks_container.on_epoch_begin(epoch, model_history)
        epoch_loss = 0

        # ensuring that the model sees samples in different order in each epoch
        permutations = torch.randperm(trn_samples)
        X_trn = X_trn[permutations]
        y_trn = y_trn[permutations]

        for batch_no in range(n_batches):
            callbacks_container.on_batch_begin(batch_no, model_history)

            # fetching data
            lower = batch_no * batch_size
            upper = min(n_samples, (batch_no + 1) * batch_size)

            x_in = X_trn[lower:upper]
            y_in = y_trn[lower:upper]

            # stepping the model
            model.zero_grad()
            class_scores = model(x_in)
            loss = loss_fn(class_scores, y_in)

            # updating the weights
            loss.backward()
            optimizer.step()

            # keeping track of the model progress
            epoch_loss += loss.item()
            train_accuracy = metrics.accuracy_score(class_scores.argmax(-1), y_in)

            model_history.append_trn_logs('loss', loss.item())
            model_history.append_trn_logs('accuracy', train_accuracy)

            callbacks_container.on_batch_end(batch_no, model_history)

        # if epoch % log_every == 0:
        model.eval()

        # computing loss/acc over the entire training / validation fold
        with torch.no_grad():
            class_scores = model(X_trn)

            val_logits = model(X_val)
            val_preds = val_logits.argmax(-1)
            val_accuracy = metrics.accuracy_score(val_preds, y_val)
            val_loss = loss_fn(val_logits, y_val).item()

        model.train()

        model_history.append_dev_logs('val_loss', val_loss)
        model_history.append_dev_logs('val_accuracy', val_accuracy)

        callbacks_container.on_epoch_end(epoch, model_history)
        if model_history.should_stop_training():
            break

    model_history.close(n_epochs)
    return model_history
