from typing import List, Optional, Any
from collections import defaultdict

import torch
import numpy as np

from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from flare.callbacks import Callback, CallbacksContainer, ProgressBar
from flare.history import ModelHistory
from flare.utilities import WrapperDataset


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

            optimizer.zero_grad()
            output = model(*batch_features)
            loss = loss_fn(output, batch_labels)
            loss.backward()
            optimizer.step()

            # All feature matrices should have the same amount of sample entries,
            # hence we can take any of them to figure out the batch size
            n_samples = batch_features[0].size(batch_index)

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
            seen_samples = 0
            for batch_id, batch_data in enumerate(val_gen):
                batch_features: list = batch_data[:-1]
                batch_labels: list = batch_data[-1]

                output = model(*batch_features)
                batch_metrics = model.metric(output, batch_labels)

                for m_name, m_value in batch_metrics.items():
                    valid_metrics['val_' + m_name] += m_value
                valid_metrics['val_loss'] += loss_fn(output, batch_labels).item()

                # All feature matrices should have the same amount of sample entries,
                # hence we can take any of them to figure out the batch size
                seen_samples += batch_features[0].size(batch_index)

            model_history.append_dev_logs(_normalize_metrics(valid_metrics, seen_samples))
        callbacks_container.on_epoch_end(epoch, model_history)
        if model_history.should_stop_training():
            break

    model_history.close(n_epochs)
    callbacks_container.on_train_end()

    return model_history


def train(model: nn.Module,
          x_trn: torch.Tensor,
          y_trn: torch.Tensor,
          loss_fn: Any,
          optimizer: Optimizer,
          n_epochs: int,
          batch_size: int,
          batch_first: bool = True,
          validation_frac: Optional[float] = None,
          x_val: Optional[torch.Tensor] = None,
          y_val: Optional[torch.Tensor] = None,
          shuffle: bool = True,
          callbacks: Optional[List[Callback]] = None) -> ModelHistory:
    """Helper function to train the model.

    # Arguments
        model: The PyTorch model
        x_trn: Tensors representing the sample features
        y_trn: Tensors with categorical labels
        loss_fn:
        optimizer:
        n_epochs:
        batch_size:
        batch_first:
        validation_frac: Percentage of the samples used for validation only
            after shuffling
        x_val:
        y_val:
        shuffle:
        callbacks:

    # Return
        A ModelHistory object with the logs of model metrics after each epoch.
    """

    if batch_size < 1:
        raise ValueError('Each batch should have at least one sample.')

    batch_index = 0 if batch_first else 1
    if isinstance(x_trn, torch.Tensor):
        n_samples = x_trn.size(batch_index)
    else:
        n_samples = x_trn[0].size(batch_index)

    dataset_val = None
    if x_val is not None or y_val is not None:
        dataset_val = WrapperDataset(x_val, y_val)
    else:
        if validation_frac is not None:
            # Need to reserve part of training data for validation
            val_samples = _ceil(n_samples * validation_frac)

            # First permute before separating the validation samples
            permutations = torch.randperm(n_samples)
            x_trn = x_trn[permutations]
            y_trn = y_trn[permutations]

            x_val = x_trn[:val_samples]
            y_val = y_trn[:val_samples]

            x_trn = x_trn[val_samples:]
            y_trn = y_trn[val_samples:]

            dataset_val = WrapperDataset(x_val, y_val)

    dataset_trn = WrapperDataset(x_trn, y_trn)
    loader_trn = DataLoader(dataset_trn, batch_size, shuffle)

    if dataset_val:
        loader_dev = DataLoader(dataset_val, batch_size, shuffle)
    else:
        loader_dev = None

    return train_on_loader(model, loader_trn, loader_dev, loss_fn, optimizer, n_epochs, batch_first, callbacks)
