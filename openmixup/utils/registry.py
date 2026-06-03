import inspect
from functools import partial

import mmcv


class Registry(object):

    def __init__(self, name):
        self._name = name
        self._module_dict = dict()

    def __repr__(self):
        format_str = self.__class__.__name__ + "(name={}, items={})".format(self._name, list(self._module_dict.keys()))
        return format_str

    @property
    def name(self):
        return self._name

    @property
    def module_dict(self):
        return self._module_dict

    def get(self, key):
        return self._module_dict.get(key, None)

    def _register_module(self, module_class, force=False):
        """Register a module.

        Args:
            module (:obj:`nn.Module`): Module to be registered.
        """
        if not inspect.isclass(module_class):
            raise TypeError("module must be a class, but got {}".format(type(module_class)))
        module_name = module_class.__name__
        if not force and module_name in self._module_dict:
            raise KeyError("{} is already registered in {}".format(module_name, self.name))
        self._module_dict[module_name] = module_class

    def register_module(self, cls=None, force=False):
        if cls is None:
            return partial(self.register_module, force=force)
        self._register_module(cls, force=force)
        return cls


def build_from_cfg(cfg, registry, default_args=None):
    """Build a module from a configuration dictionary.

    This helper looks up the class name specified by ``cfg['type']`` in the
    provided ``registry`` and instantiates it with the remaining items in ``cfg``
    (merged with ``default_args`` if supplied).

    The original implementation raised generic ``KeyError`` or ``TypeError``
    exceptions which made debugging difficult because the error messages did
    not contain the actual configuration or the list of registered classes.

    The updated version adds richer diagnostics:

    * If ``type`` is a string but not found in the registry, the error now
      includes the registry name, the missing type, and a list of the available
      keys.
    * When the class constructor raises ``TypeError`` (often due to missing
      required arguments), the error message now prints the full ``cfg`` used for
      construction as well as the resolved ``args`` after merging ``default_args``.
    * Any other unexpected exception is caught and re‑raised after printing a
      concise diagnostic.
    """
    # Basic validation
    assert isinstance(cfg, dict) and "type" in cfg, f"found config {cfg}"
    assert isinstance(default_args, dict) or default_args is None
    args = cfg.copy()
    obj_type = args.pop("type")

    # Resolve the class object
    if mmcv.is_str(obj_type):
        obj_cls = registry.get(obj_type)
        if obj_cls is None:
            available = list(registry.module_dict.keys())
            raise KeyError(f"'{obj_type}' is not registered in '{registry.name}'. Available keys: {available}.")
    elif inspect.isclass(obj_type):
        obj_cls = obj_type
    else:
        raise TypeError(f"type must be a str or a class, but got {type(obj_type)}")

    # Merge default arguments if provided
    if default_args is not None:
        for name, value in default_args.items():
            args.setdefault(name, value)

    # Attempt construction with detailed error reporting
    try:
        return obj_cls(**args)
    except TypeError as e:
        # Provide full context for missing/extra arguments
        print(f"TypeError while constructing {obj_cls.__name__} with args={args}. Error: {e}")
        raise e
    except Exception as e:
        # Catch-all for unexpected errors
        print(f"Unexpected error while building {obj_cls.__name__} from cfg={cfg} with args={args}: {e}")
        raise e
