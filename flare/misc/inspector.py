from collections import OrderedDict
from typing import Union, List, Tuple, Optional
import inspect

import torch
from torch import nn
from torch.nn.utils import rnn as rnn_utils


def summarize(model: nn.Module,
              in_tensor: Union[List[torch.Tensor], torch.Tensor],
              inspect_custom: bool = False) -> None:
    module_stack = OrderedDict()
    all_hook_handles = list()

    _filter_modules = lambda pair: pair[0][:2] != '__' and str(pair[1])[:6] == '<class'
    default_modules = set(map(
            lambda x: x[1],  # get just the module class, not its name
            filter(_filter_modules, inspect.getmembers(nn.modules))
        ))

    def get_shape(output: Union[torch.Tensor, rnn_utils.PackedSequence]):
        if isinstance(output, rnn_utils.PackedSequence):
            shape = 'PackedSequence'
        elif isinstance(output, torch.Tensor):
            # Casting torch.Shape to tuple
            shape = tuple(output.shape)
        else:
            raise RuntimeError('Unexpected module input/output type')
        return shape

    def register_hook(module):
        def hook(module: nn.Module, in_: tuple, out_: tuple):
            # module.__class__ = <class 'torch.nn.modules.linear.Linear'>
            layer_idx = len(module_stack)
            layer_name = str(module.__class__).split('.')[-1].split("'")[0]
            layer_name = f'{layer_idx}_{layer_name}'

            if isinstance(in_, (tuple, list)):
                in_shape = list(map(get_shape, in_))
            else:
                in_shape = get_shape(in_)

            if isinstance(out_, (tuple, list)):
                out_shape = list(map(get_shape, out_))
            else:
                out_shape = [get_shape(out_)]

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

    # model.apply(register_hook)
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
            str(in_shapes[0]),
            str(out_shapes[0])
        ))

        longest_seq = max(len(in_shapes), len(out_shapes))
        for seq_idx in range(1, longest_seq):
            left_padding = [' ' * 45]
            if seq_idx < len(in_shapes):
                fmt_in = f'{str(in_shapes[seq_idx]):25} '
            else:
                fmt_in = ' ' * 26
            left_padding.append(fmt_in)

            if seq_idx < len(out_shapes):
                left_padding.append(f'{str(out_shapes[seq_idx]):25}')

            print(''.join(left_padding))
    print('=' * 90)

    total_train = sum([layer['trainable'] for layer in module_stack.values()])
    total_nontrain = sum([layer['non_trainable'] for layer in module_stack.values()])
    print(f'Total params: {total_train + total_nontrain:,}')
    print(f'Trainable params: {total_train:,}')
    print(f'Frozen params: {total_nontrain:,}')
    print('_' * 90)
