# Template init file that loads everything from the subdirectory
# Good for registry based module switching system

import glob
from easyvolcap.utils.console_utils import *
from os.path import dirname, basename, join


def import_class(path: str):
    items = path.split('.')
    name = '.'.join(items[:-1])
    module = items[-1]
    return getattr(__import__(name, fromlist=[module]), module)


def import_submodules(__file__):  # note that here __file__ is passed in as an argument instead of being used as a global variable
    modules = glob.glob(join(dirname(__file__), "*"))
    __all__ = [basename(f)[:-3] if basename(f).endswith('.py') else basename(f)
               for f in modules if
               not f.endswith('__init__.py') and
               not f.endswith('.pyc') and
               not f.endswith('__pycache__')
               ]
    return __all__


def discover_modules():
    # This function imports the entry points & registers modules
    from easyvolcap import engine

    # https://rules.sonarsource.com/python/RSPEC-2208
    # from easyvolcap import * # recursively import all modules
    __import__('easyvolcap', fromlist=['*'])  # FIXME: the registry system is inherantly flawed, maybe use full paths?
    # from easyvolcap.utils.console_utils import *

    # from easyvolcap.utils.console_utils import log
    # from easyvolcap.engine.registry import Registry
    # engine_variables = {k: getattr(engine, k) for k in dir(engine)}
    # registries = {k: [k for k in v._module_dict] for k, v in engine_variables.items() if isinstance(v, Registry)}
    # log('Registered easyvolcap modules: ', registries)


@catch_throw
def prepare_shtab_parser():
    import sys
    import shtab
    import argparse
    sys.argv = ['evc', 'exp_name=preparing_shtab_parser', 'preparing_parser=True']
    from easyvolcap.engine import cfg

    parser = argparse.ArgumentParser(prog='evc', description='EasyVolumetricVideo Codebase Entrypoint')
    parser.add_argument('-c', '--config', type=str, default="configs/base.yaml", help='config file path').complete = shtab.FILE
    parser.add_argument('-t', "--type", type=str, choices=['train', 'test', 'gui'], default="train", help='execution mode, train, test or gui')  # evalute, visualize, network, dataset? (only valid when run)

    def walk_cfg(cfg=cfg._cfg_dict, prefix=''):
        for k, v in cfg.items():
            if isinstance(v, dict):
                walk_cfg(v, prefix=f'{prefix}{k}.')
            else:
                if k.endswith('configs'):
                    parser.add_argument(f'--{prefix}{k}', default=v).complete = shtab.FILE
                else:
                    parser.add_argument(f'--{prefix}{k}', default=v)
    walk_cfg()
    return parser


if __name__ == '__main__':
    parser = prepare_shtab_parser()
    args, argv = parser.parse_known_args()  # commandline arguments
    print(args)
    print(argv)
