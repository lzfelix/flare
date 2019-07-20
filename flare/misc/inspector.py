from collections import OrderedDict
from typing import Union, List

import torch
from torch import nn
from torch.nn.utils import rnn as rnn_utils


def summarize(model: nn.Module, in_tensor: Union[List[torch.Tensor], torch.Tensor]) -> None:
    module_stack = OrderedDict()
    all_hook_handles = list()

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

            # Modules usually have 1 input and 1 or more outputs
            in_shape = get_shape(in_[0])

            if isinstance(out_, (tuple, list)):
                out_shape = list(map(get_shape, out_))
            else:
                out_shape = get_shape(out_)

            trainable_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            non_trainable = sum(p.numel() for p in module.parameters() if not p.requires_grad)
            module_stack[layer_name] = {
                'in_shape': in_shape,
                'out_shape': out_shape,
                'trainable': trainable_params,
                'non_trainable': non_trainable
            }

        if not module == model:
            hook_handle = module.register_forward_hook(hook)
            all_hook_handles.append(hook_handle)

    def format_out_shape(plist):
        prefix = '\n' + ' ' * 71
        if isinstance(plist, list):
            return prefix.join([str(x) for x in plist])
        else:
            return str(plist)

    model.apply(register_hook)
    with torch.no_grad():
        model(*in_tensor)

    # Removing hooks from the original model
    for handle in all_hook_handles:
        handle.remove()

    # Formatting the output
    print('=' * 90)
    print('{:20}  {:22} {:25} {:15}'.format('Layer type',
                                            'Trainable / Frozen',
                                            'Input shape',
                                            'Output shape'))
    print('-' * 90)
    for name, specs in module_stack.items():
        line_new = "{:20}  {:10} {:10}  {:25} {:15}".format(
            name,
            str(specs['trainable']),
            str(specs['non_trainable']),
            str(specs['in_shape']),
            format_out_shape(specs['out_shape'])
        )
        print(line_new)
    print('=' * 90)

    total_train = sum([layer['trainable'] for layer in module_stack.values()])
    total_nontrain = sum([layer['non_trainable'] for layer in module_stack.values()])
    print(f'Total params: {total_train + total_nontrain:,}')
    print(f'Trainable params: {total_train:,}')
    print(f'Frozen params: {total_nontrain:,}')
    print('=' * 90)
