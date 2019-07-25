import sys
import inspect
from collections import OrderedDict
from typing import Union, List

import torch
from torch import nn
from torch.nn.utils import rnn as rnn_utils


def _compute_shape(tensor, batch_first: bool) -> List[str]:

    def get_shape(tensor: Union[torch.Tensor, rnn_utils.PackedSequence]):
        if isinstance(tensor, rnn_utils.PackedSequence):
            t, _ = rnn_utils.pad_packed_sequence(tensor, batch_first=batch_first)
            shape = str(tuple(t.shape)) + '*'
        elif isinstance(tensor, torch.Tensor):
            shape = str(tuple(tensor.shape))
        else:
            raise RuntimeError('Unexpected module input/output type')
        return shape

    if isinstance(tensor, (tuple, list)):
        tensor_shapes = list(map(get_shape, tensor))
    else:
        tensor_shapes = [get_shape(tensor)]

    return tensor_shapes


def summarize(model: nn.Module,
              in_tensor: Union[List[torch.Tensor], torch.Tensor],
              batch_first: bool = True,
              inspect_custom: bool = False) -> None:
    # TODO: Do not count repeated layer parameters twice
    """Prints the model layout.

    # Arguments
        model: The model to be summarized.
        in_tensor: The input expected by the model. This function will
            perform a single forward pass in the model without storing
            the gradients
        batch_first: Used to devise PackedSequence shapes in RNN-based
            models.
        inspect_custom: If set to True this function will list all the
            internal modules of non-standard PyTorch modules. By default
            only the custom module name is listed, along with all of its
            trainable/frozen parameters.
    """
    module_stack = OrderedDict()
    all_hook_handles = list()

    # Figures out all pyTorch default Modules, which are nn subpackages
    filter_modules = lambda pair: pair[0][:2] != '__' and str(pair[1])[:6] == '<class'
    default_modules = set(map(
            lambda x: x[1],  # get just the module class, not its name
            filter(filter_modules, inspect.getmembers(nn.modules))
        ))

    def register_hook(module):
        def hook(module: nn.Module, in_: tuple, out_: tuple):
            # module.__class__ = <class 'torch.nn.modules.linear.Linear'>
            layer_idx = len(module_stack)
            layer_name = str(module.__class__).split('.')[-1].split("'")[0]
            layer_name = f'{layer_idx}_{layer_name}'

            in_shape = _compute_shape(in_, batch_first)
            out_shape = _compute_shape(out_, batch_first)

            trainable_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            non_trainable = sum(p.numel() for p in module.parameters() if not p.requires_grad)
            module_stack[layer_name] = {
                'in_shape': in_shape,
                'out_shape': out_shape,
                'trainable': trainable_params,
                'non_trainable': non_trainable
            }

        hook_handle = module.register_forward_hook(hook)
        all_hook_handles.append(hook_handle)

    for child_module in model.children():
        if child_module == model:
            continue
        if not inspect_custom and child_module.__class__ not in default_modules:
            register_hook(child_module)
        else:
            child_module.apply(register_hook)

    with torch.no_grad():
        model(*in_tensor)

    # Removing hooks from the original model
    for handle in all_hook_handles:
        handle.remove()

    # Formatting the output
    print('_' * 90)
    print('{:20}  {:22} {:25} {:15}'.format('Layer type',
                                            'Trainable / Frozen',
                                            'Input shape',
                                            'Output shape'))
    print('=' * 90)
    for i, (name, specs) in enumerate(module_stack.items()):
        in_shapes = specs['in_shape']
        out_shapes = specs['out_shape']

        print('{:20}  {:10} {:10}  {:25} {:15}'.format(
            name,
            str(specs['trainable']),
            str(specs['non_trainable']),
            in_shapes[0],
            out_shapes[0]
        ))

        longest_seq = max(len(in_shapes), len(out_shapes))
        for seq_idx in range(1, longest_seq):
            extra_line = [' ' * 45]
            if seq_idx < len(in_shapes):
                fmt_in = f'{in_shapes[seq_idx]:25} '
            else:
                fmt_in = ' ' * 26
            extra_line.append(fmt_in)

            if seq_idx < len(out_shapes):
                extra_line.append(f'{out_shapes[seq_idx]:25}')

            print(''.join(extra_line))

        if i < len(module_stack) - 1:
            print('-' * 90)
    print('=' * 90)

    total_train = sum([layer['trainable'] for layer in module_stack.values()])
    total_nontrain = sum([layer['non_trainable'] for layer in module_stack.values()])
    print(f'Total params: {total_train + total_nontrain:,}')
    print(f'Trainable params: {total_train:,}')
    print(f'Frozen params: {total_nontrain:,}')
    print('\n(?, ?, ?)* = Unpacked sequence shape')
    print('_' * 90)
    sys.stdout.flush()
