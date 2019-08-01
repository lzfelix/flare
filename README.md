# Flare
_:fire: Flare - Going faster with pyTorch_

[![License: MIT](https://img.shields.io/badge/License-MIT-brightgreen.svg)](https://github.com/lzfelix/flare/blob/master/LICENSE)


PyTorch is a very nice framework, since it allows us to have complete control over neural models
due to its eager execution. This allows one to easily change computational paths dinamically,
diferentlly from what is done in TensorFlow and in Keras. However, developing a PyTorch model
comes at expense of writing the training loop for every program. Despite allowing the user a
greater flexibility for more intrincate models, in most of the cases this phase consists basically in
the following steps:

1. Shuffle the trainig / dev data
2. For each batch in an epoch:
    1. Make a forward pass, store the intermediate values
    2. Compute the error according to some loss function
    3. Perform the backward pass and update weights
3. Compute some metrics on the validation set
4. Go to (2) unless some criteria are met, such as:
    1. Some metric doesn't improve for e iterations;
    2. Some loss gets smaller than some value;
    3. A predefined number of epochs has elapsed


**NOTE:** I am writing this famework in my free time, so it can take some time for things starting
popping up here. Still, fell free to open issues, make PRs and discuss missing features. :smile:

## What Flare is

Writing the same loop over and over can become a waste of time and souce for subtle bugs. The
main idea of this framework is allowing you to focus on developing networks instead of writing
it, and that's all. Flare borrows several ideas form Keras for training models, so you can write
a custom PyTorch model as usual, but then call Flare's `trainer` methods to fit the network.
Training a CNN then becomes much simpler:

```python
from flare import trainer
from torch import optim

from my_networks import CustomCNN
from my_datasets import mnist

train_loader = mnist.get_dataset()
test_loader  = mnist.get_dataset(train=False)

model = CustomCNN()
opt = optim.SGD(model.parameters(), lr=1e-2, momentum=0.5)

logs = trainer.train_on_loader(model, train_loader, test_loader, F.nll_loss, opt, n_epochs=2)
```

More examples on how to incorporate Flare into your workflow in `examples/`.

## What Flare is not

Despite of borrowing ideas from Keras, this does not mean that Flare is intend to mimic it
for PyTorch, as we believe that this would obfuscate PyTorch capabilities into a wrapper. We
just want to help with the model training part. Below is a list with some Keras-like frameworks for PyTorch:

* [Poutyne](https://github.com/GRAAL-Research/poutyne) (formerly PyToune)
* [MagNet](https://github.com/MagNet-DL/magnet)

## Currently implemented features

* Training loop for `DataLoader`
* Training loop for data in tensors
* Some initialization functions
* Callbacks: ProgressBar, Early Stopping, Checkpoint, TelegramNotifyer, Custom
* A model summary similar to Keras that works with sequential and convolutional models

## How to install

Currently, Flare can only be installed from GitHub, i.e.:

```bash
pip install git+https://github.com/lzfelix/flare
```

## How metrics are computed

* Accuracy is computed over the entire training/evaluation dataset;
* Other metrics are displayed as the average obtained in each batch (including during eval mode).

## FAQ

  1. When using JupyterLab or JupyterNotebook progress bars are not properly displayed.

> Please see: https://ipywidgets.readthedocs.io/en/stable/user_install.html. Notice that you must install the `ipywidgets` package, the `nodejs` module and `labextension`.

