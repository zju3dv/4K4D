from easyvolcap.utils.import_utils import import_submodules
__all__ = import_submodules(__file__)

for module in __all__:
    try:
        # from . import *  # the actual imports
        exec(f'from . import {module}')
    except Exception as e:
        import sys
        from easyvolcap.utils.console_utils import *

        exception_type, exception_object, exception_traceback = sys.exc_info()
        filename = exception_traceback.tb_frame.f_code.co_filename
        line_number = exception_traceback.tb_lineno

        log(yellow(f'Failed to import {red(filename)}:{line_number}, {red(type(e))}: {red_slim(e)}'))
