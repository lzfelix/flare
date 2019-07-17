from torch import nn


def init_gru_(gru_module: nn.GRU) -> nn.GRU:
    """Initializes a GRU module, regardless amount of layers and directions.

    Recurrent weights use orthogonal initialization and input weights use
    xavier_normal (glorot_normal) initialization. Biases are initialized with
    zeros.
    """
    for param_name in gru_module._get_flat_weights_names():
        param_tensor = gru_module.__getattr__(param_name)
        if 'weight' in param_name:
            tensor_height = param_tensor.size(0) // 3

            # hh = hidden/hidden; ih = input/hidden tensor
            if 'hh' in param_name:
                initializer_ = nn.init.orthogonal_
            elif 'ih' in param_name:
                initializer_ = nn.init.xavier_uniform_
            else:
                raise RuntimeError(f'Unexpected parameter named "{param_name}".')

            # Initialize each tensor (reset, update, hidden) individually
            for i in range(3):
                lower = i * tensor_height
                upper = (i + 1) * tensor_height
                initializer_(param_tensor[lower:upper])

        elif 'bias' in param_name:
            nn.init.zeros_(param_tensor)
        else:
            raise RuntimeError(f'Unexpected parameter matrix named "{param_name}"')
    return gru_module
