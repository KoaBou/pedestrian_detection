import functools
import inspect
import sys
from collections import OrderedDict

import numpy as np
import torch

class Empty(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super(Empty, self).__init__()

    def forward(self, *args, **kwargs):
        if len(args) == 1:
            return args[0]
        elif len(args) == 0:
            return None
        return args



class Sequential(torch.nn.Module):
    r"""A sequential container.
    Modules will be added to it in the order they are passed in the constructor.
    Alternatively, an ordered dict of modules can also be passed in.
    To make it easier to understand, given is a small example::
        # Example of using Sequential
        model = Sequential(
                  nn.Conv2d(1,20,5),
                  nn.ReLU(),
                  nn.Conv2d(20,64,5),
                  nn.ReLU()
                )
        # Example of using Sequential with OrderedDict
        model = Sequential(OrderedDict([
                  ('conv1', nn.Conv2d(1,20,5)),
                  ('relu1', nn.ReLU()),
                  ('conv2', nn.Conv2d(20,64,5)),
                  ('relu2', nn.ReLU())
                ]))
        
        # Example of using Sequential with kwargs(python 3.6+)
        model = Sequential(
                  conv1=nn.Conv2d(1,20,5),
                  relu1=nn.ReLU(),
                  conv2=nn.Conv2d(20,64,5),
                  relu2=nn.ReLU()
                )
    """

    def __init__(self, *args, **kwargs):
        super(Sequential, self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)
        for name, module in kwargs.items():
            if sys.version_info < (3, 6):
                raise ValueError("kwargs only supported in py36+")
            if name in self._modules:
                raise ValueError("name exists.")
            self.add_module(name, module)

    def __getitem__(self, idx):
        if not (-len(self) <= idx < len(self)):
            raise IndexError('index {} is out of range'.format(idx))
        if idx < 0:
            idx += len(self)
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __len__(self):
        return len(self._modules)

    def add(self, module, name=None):
        if name is None:
            name = str(len(self._modules))
            if name in self._modules:
                raise KeyError("name exists")
        self.add_module(name, module)

    def forward(self, input):
        for module in self._modules.values():
            input = module(input)
        return input

def get_pos_to_kw_map(func):
    pos_to_kw = {}
    fsig = inspect.signature(func)
    pos = 0
    for name, info in fsig.parameters.items():
        if info.kind is info.POSITIONAL_OR_KEYWORD:
            pos_to_kw[pos] = name
        pos += 1
    return pos_to_kw


def get_kw_to_default_map(func):
    kw_to_default = {}
    fsig = inspect.signature(func)
    for name, info in fsig.parameters.items():
        if info.kind is info.POSITIONAL_OR_KEYWORD:
            if info.default is not info.empty:
                kw_to_default[name] = info.default
    return kw_to_default


def change_default_args(**kwargs):
    def layer_wrapper(layer_class):
        class DefaultArgLayer(layer_class):
            def __init__(self, *args, **kw):
                pos_to_kw = get_pos_to_kw_map(layer_class.__init__)
                kw_to_pos = {kw: pos for pos, kw in pos_to_kw.items()}
                for key, val in kwargs.items():
                    if key not in kw and kw_to_pos[key] > len(args):
                        kw[key] = val
                super().__init__(*args, **kw)

        return DefaultArgLayer

    return layer_wrapper


def np_dtype_to_torch(dtype):
    type_map = {
        np.dtype(np.float16): torch.HalfTensor,
        np.dtype(np.float32): torch.FloatTensor,
        np.dtype(np.float64): torch.DoubleTensor,
        np.dtype(np.int32): torch.IntTensor,
        np.dtype(np.int64): torch.LongTensor,
        np.dtype(np.uint8): torch.ByteTensor,
    }
    return type_map[dtype]


def np_dtype_to_np_type(dtype):
    type_map = {
        np.dtype(np.float16): np.float16,
        np.dtype(np.float32): np.float32,
        np.dtype(np.float64): np.float64,
        np.dtype(np.int32): np.int32,
        np.dtype(np.int64): np.int64,
        np.dtype(np.uint8): np.uint8,
    }
    return type_map[dtype]


def np_type_to_torch(dtype, cuda=False):
    type_map = {
        np.float16: torch.HalfTensor,
        np.float32: torch.FloatTensor,
        np.float64: torch.DoubleTensor,
        np.int32: torch.IntTensor,
        np.int64: torch.LongTensor,
        np.uint8: torch.ByteTensor,
    }
    cuda_type_map = {
        np.float16: torch.cuda.HalfTensor,
        np.float32: torch.cuda.FloatTensor,
        np.float64: torch.cuda.DoubleTensor,
        np.int32: torch.cuda.IntTensor,
        np.int64: torch.cuda.LongTensor,
        np.uint8: torch.cuda.ByteTensor,
    }
    if cuda:
        return cuda_type_map[dtype]
    else:
        return type_map[dtype]




def torch_to_np_type(ttype):
    type_map = {
        'torch.HalfTensor': np.float16,
        'torch.FloatTensor': np.float32,
        'torch.DoubleTensor': np.float64,
        'torch.IntTensor': np.int32,
        'torch.LongTensor': np.int64,
        'torch.ByteTensor': np.uint8,
        'torch.cuda.HalfTensor': np.float16,
        'torch.cuda.FloatTensor': np.float32,
        'torch.cuda.DoubleTensor': np.float64,
        'torch.cuda.IntTensor': np.int32,
        'torch.cuda.LongTensor': np.int64,
        'torch.cuda.ByteTensor': np.uint8,
    }
    return type_map[ttype]


def _torch_string_type_to_class(ttype):
    type_map = {
        'torch.HalfTensor': torch.HalfTensor,
        'torch.FloatTensor': torch.FloatTensor,
        'torch.DoubleTensor': torch.DoubleTensor,
        'torch.IntTensor': torch.IntTensor,
        'torch.LongTensor': torch.LongTensor,
        'torch.ByteTensor': torch.ByteTensor,
        'torch.cuda.HalfTensor': torch.cuda.HalfTensor,
        'torch.cuda.FloatTensor': torch.cuda.FloatTensor,
        'torch.cuda.DoubleTensor': torch.cuda.DoubleTensor,
        'torch.cuda.IntTensor': torch.cuda.IntTensor,
        'torch.cuda.LongTensor': torch.cuda.LongTensor,
        'torch.cuda.ByteTensor': torch.cuda.ByteTensor,
    }
    return type_map[ttype]


def torch_to_np_dtype(ttype):
    type_map = {
        torch.HalfTensor: np.dtype(np.float16),
        torch.FloatTensor: np.dtype(np.float32),
        torch.DoubleTensor: np.dtype(np.float64),
        torch.IntTensor: np.dtype(np.int32),
        torch.LongTensor: np.dtype(np.int64),
        torch.ByteTensor: np.dtype(np.uint8),
    }
    return type_map[ttype]


def isinf(tensor):
    return tensor == torch.FloatTensor([float('inf')]).type_as(tensor)


def to_tensor(arg):
    if isinstance(arg, np.ndarray):
        return torch.from_numpy(arg).type(np_dtype_to_torch(arg.dtype))
    elif isinstance(arg, (list, tuple)):
        arg = np.array(arg)
        return torch.from_numpy(arg).type(np_dtype_to_torch(arg.dtype))
    else:
        raise ValueError("unsupported arg type.")


def zeros(*sizes, dtype=np.float32, cuda=False):
    torch_tensor_cls = np_type_to_torch(dtype, cuda)
    return torch_tensor_cls(*sizes).zero_()


def get_tensor_class(tensor):
    return _torch_string_type_to_class(tensor.type())