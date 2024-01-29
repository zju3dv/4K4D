# The eigine: configuration system, module registration system are borrowed from mmcv
# https://github.com/open-mmlab/mmcv
# registry.py: https://raw.githubusercontent.com/open-mmlab/mmcv/master/mmcv/utils/registry.py
# config.py: https://raw.githubusercontent.com/open-mmlab/mmcv/master/mmcv/utils/config.py

# NOTE: one key difference is that we allow duplicated keys in base configurations

# These two files depends on misc.py and path.py for the core functionality to be invoked
# misc.py: https://raw.githubusercontent.com/open-mmlab/mmcv/master/mmcv/utils/misc.py
# path.py: https://raw.githubusercontent.com/open-mmlab/mmcv/master/mmcv/utils/path.py

# Other downloaded files from mmcv: https://github.com/open-mmlab/mmcv/tree/master/mmcv/fileio

# Training structure:
# - Construct dataset
# - Construct model (calls sampler -> network -> renderer -> supervisor)
#   - Construct sampler, renderer, network (core logic) and supervisor (loss logic)
# - Construct runner (calls dataset -> model -> optimizer -> scheduler)
# - Constuct visualizer (calls dataset -> model when appropriate)

# Visualizing structure:
# - Construct dataset
# - Construct model (calls sampler -> network -> renderer -> supervisor)
#   - Construct sampler, renderer, network (core logic) and supervisor (loss logic)
# - Constuct visualizer (calls dataset -> model)
import argparse

# No circular imports for these boys!
from easyvolcap.engine.registry import Registry, call_from_cfg, callable_from_cfg  # will cause circular import if using full path here
from easyvolcap.engine.config import Config, DictAction, ConfigDict
from easyvolcap.utils.console_utils import *

# Here we predefine a list of valid registers to be used in the construction and expansion of this repos
VISUALIZERS = Registry('visualizers')  # MARK: (constantly changed)
EVALUATORS = Registry('evaluators')  # (rarely changed)

DATASETS = Registry('datasets')  # MARK: (constantly changed)
DATALOADERS = Registry('dataloaders')  # (rarely changed)
DATASAMPLERS = Registry('datasamplers')  # samples rays on multi-view videos # MARK: (constantly changed)

MODELS = Registry('models')  # contains model logic (rarely changed)
CAMERAS = Registry('cameras')  # contains model logic (rarely changed)
NETWORKS = Registry('networks')  # contains most optimizable parameters # MARK: (constantly changed)
EMBEDDERS = Registry('embedders')  # components for networks # MARK: (constantly changed)
REGRESSORS = Registry('regressors')  # components for networks # MARK: (constantly changed)
SAMPLERS = Registry('samplers')  # sample points on rays -> holds pointer to network # MARK: (constantly changed)
RENDERERS = Registry('renderers')  # integrate points to rays -> holds pointer to network # MARK: (constantly changed)
SUPERVISORS = Registry('supervisors')  # for computing loss -> holds pointer to network # MARK: (constantly changed)

RUNNERS = Registry('runners')  # contains runner logic (sometimes changed)
OPTIMIZERS = Registry('optimizers')  # (rarely changed)
SCHEDULERS = Registry('schedulers')  # (rarely changed)
MODERATORS = Registry('moderators')  # (rarely changed)
RECORDERS = Registry('recorders')  # (rarely changed)

# NOTE that if you're trying to build something, please import cfg to entry the program entry point, I know this sounds janky but it's probably the least janky way to do it for now
# One good thing is they will error out if you did't import those


# This is a convention used in NeuralBody based configuration system
# I find it important to make the configuration object accessible from anywhere (thus for some small functions, you don't need to pass the config object around)
# Although for key components like datasets, models and runners, we'd still use the config object for initialization

# This file is the effective entry point of the code
# We'd like the cfg object to be accessible from anywhere, thus this file exists and anyone can import from it

@catch_throw
def get_parser():
    parser = argparse.ArgumentParser(prog='evc', description='EasyVolcap Project')
    parser.add_argument('-c', '--config', type=str, default="", help='config file path')
    parser.add_argument('-t', "--type", type=str, choices=['train', 'test', 'gui'], default="train", help='execution mode, train, test or gui')  # evalute, visualize, network, dataset? (only valid when run)
    parser.add_argument("opts", action=DictAction, nargs=argparse.REMAINDER)
    return parser


@catch_throw
def update_cfg(cfg: Config):
    # Here, we define some logics for updating the config object
    # i.e. when experiment name is given, the output directory is automatically updated
    # cfg.record_dir = join(cfg.record_dir, cfg.exp_name)  # tensorboard logs
    # cfg.result_dir = join(cfg.result_dir, cfg.exp_name)  # general purpose results (evaluation)
    # cfg.trained_model_dir = join(cfg.trained_model_dir, cfg.exp_name)  # trained_models (chkpts)
    # NOTE: these kind of global variables should be kept to a minimum
    return cfg  # although this return is not fully needed


@catch_throw
def parse_cfg(args):
    args.config = args.config.split(',')  # maybe the user used commas
    configs = args.config[1:]  # other files are considered as base files
    args.config = args.config[0]  # at least one config file is required
    if 'configs' in args.opts:
        if isinstance(args.opts['configs'], list): args.opts['configs'] += configs
        else: args.opts['configs'] = [args.opts['configs']] + configs
    else: args.opts['configs'] = configs

    if exists(args.config):
        cfg = Config.fromfile(args.config)  # load external configuration file (with hierarchy)
        cfg.merge_from_dict(args.opts)  # load commandline arguments
        cfg = update_cfg(cfg)
        return cfg
    elif not args.config:  # ''
        # Default config object
        return Config(
            dotdict(
                exp_name='base',
                dataloader_cfg=dotdict(dataset_cfg=dotdict()),
                runner_cfg=dotdict(ep_iter=500, epochs=400, visualizer_cfg=dotdict(save_tag='', result_dir='')),
                viewer_cfg=dotdict(type='VolumetricVideoViewer'),
                fix_random=False,
                allow_tf32=True,
                deterministic=False,
                benchmark=False,
                mocking=True,
            )
        )  # empty config
    else:
        raise FileNotFoundError(f"Config file {args.config} not found")
        # raise FileNotFoundError(f"Config file {markup_to_ansi(blue(args.config))} not found")


parser = get_parser()
args, argv = parser.parse_known_args()  # commandline arguments
argv = [v.strip('-') for v in argv]  # arguments starting with -- will not automatically go to the ops dict, need to parse them again
argv = parser.parse_args(argv)  # the reason for -- arguments is that almost all shell completion requires a prefix for optional arguments
args.opts.update(argv.opts)
cfg = parse_cfg(args)
