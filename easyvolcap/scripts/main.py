# fmt: off
from easyvolcap.utils.console_utils import *
import torch  # make early keyboard interrupt possible
import torch.distributed as dist

from typing import Callable
from torch.nn.parallel import DistributedDataParallel as DDP

from easyvolcap.engine import args, cfg  # commandline entrypoint
from easyvolcap.engine import RUNNERS, MODELS, DATALOADERS
from easyvolcap.engine import callable_from_cfg, call_from_cfg

from easyvolcap.utils.import_utils import discover_modules
discover_modules() # will launch through this interface

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from easyvolcap.dataloaders.datasamplers import SequentialSampler, DistributedSampler, BatchSampler
    from easyvolcap.dataloaders.datasets.volumetric_video_dataset import VolumetricVideoDataset
    from easyvolcap.dataloaders.volumetric_video_dataloader import VolumetricVideoDataloader
    from easyvolcap.models.volumetric_video_model import VolumetricVideoModel
    from easyvolcap.runners.volumetric_video_runner import VolumetricVideoRunner
    from easyvolcap.runners.volumetric_video_viewer import VolumetricVideoViewer

from easyvolcap.utils.data_utils import DataSplit
from easyvolcap.utils.base_utils import dotdict
from easyvolcap.utils.dist_utils import synchronize, get_rank, get_distributed
from easyvolcap.utils.net_utils import setup_deterministic, number_of_params
from easyvolcap.utils.prof_utils import setup_profiler, profiler_start, profiler_stop
# fmt: on


def launcher(runner_function: Callable,  # viewer.run or runner.train or runner.test
             runner_object: "VolumetricVideoRunner" = None,
             exp_name='nerft',
             detect_anomaly: bool = False,
             profiler_cfg: dotdict = dotdict(),  # for debug use only

             *args,
             **kwargs
             ):

    # Perform the actual training
    setup_profiler(**profiler_cfg)  # MARK: overwritting default configs
    prev_anomaly = torch.is_anomaly_enabled()
    torch.set_anomaly_enabled(detect_anomaly)
    profiler_start()  # already setup

    # Give the user some time to save states
    log('Launching runner for experiment:', magenta(exp_name))
    cfg.runner = runner_object  # holds a global reference for hacky usage # MARK: GLOBAL
    runner_function()

    profiler_stop()  # already setup
    torch.set_anomaly_enabled(prev_anomaly)


def preflight(
    fix_random: bool = False,
    allow_tf32: bool = True,
    deterministic: bool = False,  # for debug use only
    benchmark: Union[bool, str] = True,  # for static sized input
    ignore_breakpoint: bool = False,
    hide_progress: bool = False,
    less_verbose: bool = False,
    hide_output: bool = False,
    **kwargs,
):
    # Some early on GUI specific configurations
    if ignore_breakpoint: disable_breakpoint()
    if hide_progress: disable_progress()
    if hide_output: disable_console()
    if less_verbose: disable_verbose_log()
    if benchmark == 'train': benchmark = args.type == 'train'  # for static sized input

    # Maybe make this run deterministic?
    setup_deterministic(fix_random, allow_tf32, deterministic, benchmark)  # whether to be deterministic throughout the whole training process?

    # Log the experiment name for later usage
    log(f"Starting experiment: {magenta(cfg.exp_name)}, command: {magenta(args.type)}")  # MARK: GLOBAL


@catch_throw
@callable_from_cfg
def gui(
    viewer_cfg: dotdict = dotdict(type="VolumetricVideoViewer"),  # use different naming for config here, is this good?
    invokation_type: str = 'test',  # TODO: implement camera and other dataset types

    # Reproducibility configuration
    base_device: str = 'cuda',
    dry_run: bool = False,  # return without hassle
    **kwargs,
):
    # # Maybe make this run deterministic?
    # preflight(**kwargs)  # whether to be deterministic throughout the whole training process?

    # Need to config all configurable options from here (although some of the building logic makes this hard)
    # It's better to refrain people from manually building all gui components
    # However, the simplest form should only be a viewer instead of complecated gui trainer, trainer should run in commandline

    # Basic steps for the gui
    # Build the camera model (main input for everything else), this is actually the user controlled camera parameters (including t)
    #     Everything else should be configured through the attributes of the model (more likely dubbed render_options?)
    #     This is the only thing different from the training or inference loop, where the camera parameters are loaded
    #     This class should be built later than dataset, since we need to extract some init camera params from the dataset for placement
    # Build the dataset (not dataloader since no preloading or caching is needed), this accepts camera inputs
    # Build the model (full model, without any hustle), this accepts dataset input
    # Build the viewer, interacts with the screen, dataset and model
    #     Every other components built should be passed into this viewer
    #     Maybe, just maybe we could launch multiple threads for rendering and training?
    #     Like instanciate a new runner for training with the same model and render something out once in a while?
    #     Note that the viewer should also be a type of runner, we need to write it in a similar way to trainer or tester
    #     The camera class could only get K, R, T, and the rest like height or width are directly read from the viewer?
    #     * Should we just integrate the camera inside the viewer?
    runner: "VolumetricVideoRunner" = globals()[invokation_type](kwargs,
                                                                 base_device=base_device,
                                                                 dry_run=True,
                                                                 )  # return the runner (trainer) immediately
    viewer: "VolumetricVideoViewer" = RUNNERS.build(viewer_cfg, runner=runner)  # will start the window
    if dry_run: return runner  # just construct everything, then return

    launcher(**kwargs, runner_function=viewer.run, runner_object=runner)


@catch_throw
@callable_from_cfg
def test(
    model_cfg: dotdict = dotdict(type="VolumetricVideoModel"),
    val_dataloader_cfg: dotdict = dotdict(
        type="VolumetricVideoDataloader",
        max_iter=-1,
        sampler_cfg=dotdict(
            type="SequentialSampler",  # changed type
        ),
        dataset_cfg=dotdict(
            type="VolumetricVideoDataset",  # TODO: not overwritting, repeated
            split=DataSplit.VAL.name,
            ratio=0.25,  # faster visualization
        ),
    ),
    runner_cfg: dotdict = dotdict(type="VolumetricVideoRunner",
                                  optimizer_cfg=dotdict(type=None),
                                  scheduler_cfg=dotdict(type=None),
                                  ),

    # Reproducibility configuration
    base_device: str = 'cuda',

    record_images_to_tb: bool = False,  # MARK: insider config # this is slow
    print_test_progress: bool = True,  # MARK: insider config # this is slow
    dry_run: bool = False,
    **kwargs,
):
    # Maybe make this run deterministic?
    preflight(**kwargs)  # whether to be deterministic throughout the whole training process?

    # Construct other parts of the training process
    val_dataloader: "VolumetricVideoDataloader" = DATALOADERS.build(val_dataloader_cfg)  # reuse the validataion

    model: "VolumetricVideoModel" = MODELS.build(model_cfg)
    model = model.to(base_device, non_blocking=True)

    runner: "VolumetricVideoRunner" = RUNNERS.build(runner_cfg,
                                                    model=model,
                                                    dataloader=None,  # no training dataloader
                                                    test_only=True,  # no training
                                                    record_images_to_tb=record_images_to_tb,  # another default
                                                    print_test_progress=print_test_progress,  # another default
                                                    val_dataloader=val_dataloader)

    if dry_run: return runner  # just construct everything, then return

    # Just run, no gossip
    launcher(**kwargs, runner_function=runner.test, runner_object=runner)


@catch_throw
@callable_from_cfg
def train(
    model_cfg: dotdict = dotdict(type="VolumetricVideoModel"),
    dataloader_cfg: dotdict = dotdict(type="VolumetricVideoDataloader"),
    val_dataloader_cfg: dotdict = dotdict(
        type="VolumetricVideoDataloader",
        max_iter=-1,  # awkward
        # num_workers=0,  # if multiprocessing, will mess up normal dataloader
        sampler_cfg=dotdict(
            type="SequentialSampler",  # changed type
        ),
        dataset_cfg=dotdict(
            type="VolumetricVideoDataset",  # TODO: not overwritting, repeated
            split=DataSplit.VAL.name,
            ratio=0.25,  # faster visualization
        ),
        batch_sampler_cfg=dotdict(type="BatchSampler",
                                  batch_size=1),
    ),
    runner_cfg: dotdict = dotdict(type="VolumetricVideoRunner"),

    # Distributed training
    distributed: bool = False,
    find_unused_parameters: bool = False,

    # Printing configuration
    dry_run: bool = False,  # only print network and exit
    print_model: bool = False,  # since the network is pretty complex, give the option to print

    # Reproducibility configuration
    base_device: str = 'cuda',
    **kwargs,
):
    # Maybe make this run deterministic?
    preflight(**kwargs)  # whether to be deterministic throughout the whole training process?

    # Construct other parts of the training process
    dataloader: "VolumetricVideoDataloader" = DATALOADERS.build(dataloader_cfg)
    if not get_rank(): val_dataloader: "VolumetricVideoDataloader" = DATALOADERS.build(val_dataloader_cfg)

    # Handle distributed training
    if distributed:
        dist.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    # Handle model placement
    rank = get_rank()
    device = torch.device(f'{base_device}:{rank}')
    torch.cuda.set_device(device)

    # Model building and distributed training related stuff
    model: "VolumetricVideoModel" = MODELS.build(model_cfg)  # some components are directly moved to cuda when building
    model.to(device, non_blocking=True)  # move this model to this specific device

    # Handle distributed model
    if get_distributed():
        model = DDP(
            model,
            device_ids=[rank],
            output_device=rank,
            find_unused_parameters=find_unused_parameters,
        )

    # Construct the runner (optimization loop controller)
    runner: "VolumetricVideoRunner" = RUNNERS.build(runner_cfg,
                                                    model=model,
                                                    dataloader=dataloader,
                                                    val_dataloader=val_dataloader if not get_rank() else None)

    if print_model and not get_rank():  # only print once
        # For some methods, both the network and the sampler or even the renderer contains optimizable parameters
        # But the sampler and render both has a reference to the network, which gets printed (not saved, tested)
        pprint(model)  # with indent guides
    try:
        nop = number_of_params(model)
        log(f'Number of optimizable parameters: {nop} ({nop / 1e6:.2f} M)')
    except ValueError as e:
        # Ignore: Attempted to use an uninitialized parameter in <method 'numel' of 'torch._C._TensorBase' objects>
        pass

    if dry_run: return runner  # just construct everything, then return

    # The actual calling, with grace full exit
    launcher(**kwargs, runner_function=runner.train, runner_object=runner)


def main():
    if cfg.mocking: log(f'Modules imported. No config loaded for: {yellow(args.type)}, pass config file using `-c <PATH_TO_CONFIG>`')  # MARK: GLOBAL
    else: globals()[args.type](cfg)  # invoke this (call callable_from_cfg -> call_from_cfg)


# Module name == '__main__', this is the outermost commandline entry point
if __name__ == '__main__':
    main()
