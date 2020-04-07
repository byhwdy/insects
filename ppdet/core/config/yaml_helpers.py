import importlib
import inspect

import yaml
from .schema import SharedConfig

__all__ = ['serializable', 'Callable']


def _make_python_constructor(cls):
    def python_constructor(loader, node):
        if isinstance(node, yaml.SequenceNode):
            args = loader.construct_sequence(node, deep=True)
            return cls(*args)
        else:
            kwargs = loader.construct_mapping(node, deep=True)
            try:
                return cls(**kwargs)
            except Exception as ex:
                print("Error when construct {} instance from yaml config".
                      format(cls.__name__))
                raise ex

    return python_constructor

def serializable(cls):
    """
    Add loader and dumper for given class, which must be
    "trivially serializable"

    Args:
        cls: class to be serialized

    Returns: cls
    """
    yaml.add_constructor(u'!{}'.format(cls.__name__),
                         _make_python_constructor(cls))
    return cls
    