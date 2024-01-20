from easyvolcap.utils.import_utils import import_submodules
__all__ = import_submodules(__file__)

for module in __all__:
    try:
        # from . import *  # the actual imports
        exec(f'from . import {module}')
    except ModuleNotFoundError as e:
        from easyvolcap.utils.console_utils import *
        log(yellow(f'Failed to import {red(module)}{red(".py")}, missing package: {red(e.name)}'))
    except Exception as e:
        from easyvolcap.utils.console_utils import *
        stacktrace()  # print a full stacktrace for the users convenience
