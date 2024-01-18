from __future__ import annotations
from copy import copy
from typing import Mapping, TypeVar, Union, Iterable, Callable, Dict, List
# these are generic type vars to tell mapping to accept any type vars when creating a type
KT = TypeVar("KT")  # key type
VT = TypeVar("VT")  # value type

# TODO: move this to engine implementation
# TODO: this is a special type just like Config
# ? However, dotdict is a general purpose data passing object, instead of just designed for config
# The only reason we defined those special variables are for type annotations
# If removed, all will still work flawlessly, just no editor annotation for output, type and meta


def return_dotdict(func: Callable):
    def inner(*args, **kwargs):
        return dotdict(func(*args, **kwargs))
    return inner


class DoNothing:
    def __getattr__(self, name):
        def method(*args, **kwargs):
            pass
        return method


class dotdict(dict, Dict[KT, VT]):
    """
    This is the default data passing object used throughout the codebase
    Main function: dot access for dict values & dict like merging and updates

    a dictionary that supports dot notation 
    as well as dictionary access notation 
    usage: d = make_dotdict() or d = make_dotdict{'val1':'first'})
    set attributes: d.val2 = 'second' or d['val2'] = 'second'
    get attributes: d.val2 or d['val2']
    """

    def update(self, dct: Dict = None, **kwargs):
        dct = copy(dct)  # avoid modifying the original dict, use super's copy to avoid recursion

        # Handle different arguments
        if dct is None:
            dct = kwargs
        elif isinstance(dct, Mapping):
            dct.update(kwargs)
        else:
            super().update(dct, **kwargs)
            return

        # Recursive updates
        for k, v in dct.items():
            if k in self:

                # Handle type conversions
                target_type = type(self[k])
                if not isinstance(v, target_type):
                    # NOTE: bool('False') will be True
                    if target_type == bool and isinstance(v, str):
                        dct[k] = v == 'True'
                    else:
                        dct[k] = target_type(v)

                if isinstance(v, dict):
                    self[k].update(v)  # recursion from here
                else:
                    self[k] = v
            else:
                if isinstance(v, dict):
                    self[k] = dotdict(v)  # recursion?
                elif isinstance(v, list):
                    self[k] = [dotdict(x) if isinstance(x, dict) else x for x in v]
                else:
                    self[k] = v
        return self

    def __init__(self, *args, **kwargs):
        self.update(*args, **kwargs)

    copy = return_dotdict(dict.copy)
    fromkeys = return_dotdict(dict.fromkeys)

    # def __hash__(self):
    #     # return hash(''.join([str(self.values().__hash__())]))
    #     return super(dotdict, self).__hash__()

    # def __init__(self, *args, **kwargs):
    #     super(dotdict, self).__init__(*args, **kwargs)

    """
    Uncomment following lines and 
    comment out __getattr__ = dict.__getitem__ to get feature:
    
    returns empty numpy array for undefined keys, so that you can easily copy things around
    TODO: potential caveat, harder to trace where this is set to np.array([], dtype=np.float32)
    """

    def __getitem__(self, key):
        try:
            return dict.__getitem__(self, key)
        except KeyError as e:
            raise AttributeError(e)
    # MARK: Might encounter exception in newer version of pytorch
    # Traceback (most recent call last):
    #   File "/home/xuzhen/miniconda3/envs/torch/lib/python3.9/multiprocessing/queues.py", line 245, in _feed
    #     obj = _ForkingPickler.dumps(obj)
    #   File "/home/xuzhen/miniconda3/envs/torch/lib/python3.9/multiprocessing/reduction.py", line 51, in dumps
    #     cls(buf, protocol).dump(obj)
    # KeyError: '__getstate__'
    # MARK: Because you allow your __getattr__() implementation to raise the wrong kind of exception.
    # FIXME: not working typing hinting code
    __getattr__: Callable[..., 'torch.Tensor'] = __getitem__  # type: ignore # overidden dict.__getitem__
    __getattribute__: Callable[..., 'torch.Tensor']  # type: ignore
    # __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    # TODO: better ways to programmically define these special variables?

    @property
    def meta(self) -> dotdict:
        # Special variable used for storing cpu tensor in batch
        if 'meta' not in self:
            self.meta = dotdict()
        return self.__getitem__('meta')

    @meta.setter
    def meta(self, meta):
        self.__setitem__('meta', meta)

    @property
    def output(self) -> dotdict:  # late annotation needed for this
        # Special entry for storing output tensor in batch
        if 'output' not in self:
            self.output = dotdict()
        return self.__getitem__('output')

    @output.setter
    def output(self, output):
        self.__setitem__('output', output)

    @property
    def persistent(self) -> dotdict:  # late annotation needed for this
        # Special entry for storing persistent tensor in batch
        if 'persistent' not in self:
            self.persistent = dotdict()
        return self.__getitem__('persistent')

    @persistent.setter
    def persistent(self, persistent):
        self.__setitem__('persistent', persistent)

    @property
    def type(self) -> str:  # late annotation needed for this
        # Special entry for type based construction system
        return self.__getitem__('type')

    @type.setter
    def type(self, type):
        self.__setitem__('type', type)

    def to_dict(self):
        out = dict()
        for k, v in self.items():
            if isinstance(v, dotdict):
                v = v.to_dict()  # recursion point
            out[k] = v
        return out


class default_dotdict(dotdict):
    def __init__(self, default_type=object, *arg, **kwargs):
        super().__init__(*arg, **kwargs)
        dict.__setattr__(self, 'default_type', default_type)

    def __getitem__(self, key):
        try:
            return super().__getitem__(key)
        except (AttributeError, KeyError) as e:
            super().__setitem__(key, dict.__getattribute__(self, 'default_type')())
            return super().__getitem__(key)
