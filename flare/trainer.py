from typing import List, Optional, Any
from collections import defaultdict

import torch
import numpy as np

from torch.optim import Optimizer
from torch import nn
from torch.utils.data import DataLoader
import sklearn

from flare.callbacks import Callback, CallbacksContainer, ProgressBar
from flare.history import ModelHistory


_ceil = lambda x: int(np.ceil(x))


def _normalize_metrics(metrics: dict, seen_samples: int):
    return {m_name: m_value / seen_samples for m_name, m_value in metrics.items()}


def train_on_loader(model: nn.Module,
                    train_gen: DataLoader,
                    val_gen: Optional[DataLoader],
                    loss_fn: Any,
                    optimizer: Optimizer,
                    n_epochs: int,
                    batch_first: bool = True,
                    callbacks: Optional[List[Callback]] = None) -> ModelHistory:
    """
    Helper function to train a model using torch DataLoader.

    # Arguments
        model: The PyTorch model.
        train_gen: A DataLoader containing the training data.
        val_gen: A DataLoader containing the validation data.
        loss_fn: The loss function from which gradients are computed.
        optimizer: The optimizer used in the backpropagation step.
        n_epochs: How many passes should be performed over the train_gen.
        callbacks: List of utility callbacks to help training the model.

    # Return
        A ModelHistory object representing the model training history.
    """

    callbacks_container = CallbacksContainer(callbacks or [])
    callbacks_container.append(ProgressBar(len(train_gen), n_epochs))
    batch_index = 0 if batch_first else 1

    model_history = ModelHistory(model)
    for epoch in range(1, n_epochs + 1):
        model.train()
        callbacks_container.on_epoch_begin(epoch, model_history)

        epoch_loss = 0
        seen_samples = 0
        training_metrics = defaultdict(int)

        for batch_id, batch_data in enumerate(train_gen):
            callbacks_container.on_batch_begin(batch_id, model_history)

            # even if batch_data = [x, y], batch_features = [x] and batch_y = [y]
            batch_features: list = batch_data[:-1]
            batch_labels: list = batch_data[-1]

            # All feature matrices should have the same amount of sample entries,
            # hence we can take any of them to figure out the batch size
            n_samples = batch_features[0].size(batch_index)

            optimizer.zero_grad()
            output = model(*batch_features)
            loss = loss_fn(output, batch_labels)
            loss.backward()
            optimizer.step()

            seen_samples += n_samples
            epoch_loss += loss.item()

            # Metrics accumulator
            batch_metrics = model.metric(output, batch_labels)
            for m_name, m_value in batch_metrics.items():
                training_metrics[m_name] += m_value

            # Need to take care of the model loss separately
            training_metrics['loss'] = epoch_loss

            # model_history.append_batch_data(training_metrics)
            model_history.append_batch_data(_normalize_metrics(training_metrics, seen_samples))

            # print(metrics)
            # metrics = {name: value / seen_samples for name, value in metrics.items()}
            callbacks_container.on_batch_end(batch_id, model_history)

        model_history.append_trn_logs(_normalize_metrics(training_metrics, seen_samples))

        # Just ignore the rest if there's no validation data
        if not val_gen:
            continue

        model.eval()
        with torch.no_grad():
            valid_metrics = defaultdict(int)
            for batch_id, batch_data in enumerate(val_gen):
                batch_features: list = batch_data[:-1]
                batch_labels: list = batch_data[-1]

                output = model(*batch_features)
                batch_metrics = model.metric(output, batch_labels)

                for m_name, m_value in batch_metrics.items():
                    valid_metrics['val_' + m_name] += m_value
                valid_metrics['val_loss'] += loss_fn(output, batch_labels).item()

            model_history.append_dev_logs(_normalize_metrics(valid_metrics, len(val_gen.dataset)))
        callbacks_container.on_epoch_end(epoch, model_history)
        if model_history.should_stop_training():
            break

    model_history.close(n_epochs)
    callbacks_container.on_train_end()

    return model_history


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

    # First dimension in the amount of samples
    n_samples = X.shape[0]
    val_samples = _ceil(n_samples * validation_frac)
    trn_samples = n_samples - val_samples

    n_batches = _ceil(trn_samples / batch_size)

    # First permute before separating the validation samples
    permutations = torch.randperm(n_samples)
    X = X[permutations]
    y = y[permutations]
    
    # Getting the validation samples
    X_val = X[:val_samples]
    y_val = y[:val_samples]
    
    # The remaining are train samples
    X_trn = X[val_samples:]
    y_trn = y[val_samples:]

    callbacks_container = CallbacksContainer(callbacks or [])
    callbacks_container.append(ProgressBar(n_batches, n_epochs))

    model_history = ModelHistory(model)
    for epoch in range(1, n_epochs + 1):
        callbacks_container.on_epoch_begin(epoch, model_history)
        epoch_loss = 0

        # Ensuring that the model sees samples in different order in each epoch
        permutations = torch.randperm(trn_samples)
        X_trn = X_trn[permutations]
        y_trn = y_trn[permutations]

        for batch_no in range(n_batches):
            callbacks_container.on_batch_begin(batch_no, model_history)

            # Fetching data
            lower = batch_no * batch_size
            upper = min(n_samples, (batch_no + 1) * batch_size)

            x_in = X_trn[lower:upper]
            y_in = y_trn[lower:upper]

            # Stepping the model
            model.zero_grad()
            class_scores = model(*x_in)
            loss = loss_fn(class_scores, y_in)

            # Updating the weights
            loss.backward()
            optimizer.step()

            # Keeping track of the model progress
            epoch_loss += loss.item()
            train_accuracy = sklearn.metrics.accuracy_score(class_scores.argmax(-1), y_in)

            model_history.append_trn_logs('loss', loss.item())
            model_history.append_trn_logs('accuracy', train_accuracy)

            callbacks_container.on_batch_end(batch_no, model_history)

        model.eval()

        # Computing loss/acc over the entire training / validation fold
        with torch.no_grad():
            val_logits = model(X_val)
            val_preds = val_logits.argmax(-1)
            val_accuracy = sklearn.metrics.accuracy_score(val_preds, y_val)
            val_loss = loss_fn(val_logits, y_val).item()

        model.train()

        model_history.append_dev_logs('val_loss', val_loss)
        model_history.append_dev_logs('val_accuracy', val_accuracy)

        callbacks_container.on_epoch_end(epoch, model_history)
        if model_history.should_stop_training():
            break

    model_history.close(n_epochs)
    callbacks_container.on_train_end()

    return model_history
