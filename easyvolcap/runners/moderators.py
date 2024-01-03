import typing
from typing import List, Tuple, Dict
if typing.TYPE_CHECKING:
    from easyvolcap.runners.volumetric_video_runner import VolumetricVideoRunner

from enum import Enum, auto
from easyvolcap.engine import MODERATORS
from easyvolcap.utils.console_utils import *


@MODERATORS.register_module()
class NoopModerator:
    def __init__(self,
                 runner: "VolumetricVideoRunner",
                 **kwargs,
                 ):
        self.runner = runner

    def state_dict(self):
        return {}

    def load_state_dict(self, state: dict):
        pass

    def step(self):
        pass


@MODERATORS.register_module()
class DatasetRatioModerator:
    def __init__(self,
                 runner: "VolumetricVideoRunner",
                 milestones: List[Tuple[int]] = [(0, 1.0)],
                 #  milestones: List[Tuple[int]] = [(0, 0.25), (500, 0.5), (1000, 1.0)],
                 skip_first: bool = False,  # this will give you an idea of the memory consumption

                 total_iter: int = 200000,  # 200k
                 **kwargs,
                 ):
        self.runner = runner
        self.milestones = milestones
        self.iter = 0

        if not skip_first:
            self.step()
            self.iter = 0

    def state_dict(self):
        return {'step': self.iter}

    def load_state_dict(self, state: dict):
        self.iter = state['step']
        self.step()

    def step(self):
        for i, r in self.milestones:
            if self.iter >= i:
                # In multi process dataset, how do we change this setting?
                if hasattr(self.runner, 'dataloader') and self.runner.dataloader is not None:
                    self.runner.dataloader.dataset.render_ratio = r
                if hasattr(self.runner, 'val_dataloader') and self.runner.val_dataloader is not None:
                    self.runner.val_dataloader.dataset.render_ratio = r
        self.iter = self.iter + 1


@MODERATORS.register_module()
class DatasetCenterCropRatioModerator(DatasetRatioModerator):
    def __init__(self,
                 runner: "VolumetricVideoRunner",
                 milestones: List[Tuple[int]] = [(0, 1.0)],
                 skip_first: bool = False,  # this will give you an idea of the memory consumption

                 total_iter: int = 500,  # 500
                 **kwargs,
                 ):
        super().__init__(runner, milestones, skip_first, total_iter, **kwargs)

    def step(self):
        for i, r in self.milestones:
            if self.iter >= i:
                # In multi process dataset, how do we change this setting?
                if hasattr(self.runner, 'dataloader') and self.runner.dataloader is not None:
                    self.runner.dataloader.dataset.render_center_crop_ratio = r
                if hasattr(self.runner, 'val_dataloader') and self.runner.val_dataloader is not None:
                    self.runner.val_dataloader.dataset.render_center_crop_ratio = r
        self.iter = self.iter + 1


@MODERATORS.register_module()
class AlternatingModerator:
    def __init__(self,
                 runner: "VolumetricVideoRunner",

                 # Pattern configs
                 pattern_cfg: dotdict = dotdict(
                     patch_size=[256, 256],
                     n_rays=65536,
                 ),

                 # Housekeeping
                 **kwargs,
                 ):
        self.iter = -1
        self.runner = runner

        # Alternator should only work on training
        self.disabled = not hasattr(self.runner.dataloader, 'dataset')
        if self.disabled: return

        # Setup pattern and pattern memory
        self.pattern = list(pattern_cfg.keys())
        self.pattern_cfg = pattern_cfg
        self.length = len(self.pattern)
        self.dataset = self.runner.dataloader.dataset
        self.memory = dotdict({p: getattr(self.dataset, p).clone() for p in self.pattern})

        # # Special care for n_rays TODO: Make this configurable
        # self.supervisor = self.runner.model.supervisor
        # self.static = dotdict(perc_loss_weight=self.supervisor.perc_loss_weight,
        #                       ssim_loss_weight=self.supervisor.ssim_loss_weight)

        # Setting things up at least once
        self.step()
        log('Setting up alternating pattern:', line(self.pattern_cfg))

    def state_dict(self):
        return {'step': self.iter}

    def load_state_dict(self, state: dict):
        self.iter = state['step']
        self.step()

    def step(self):
        self.iter = self.iter + 1
        if self.disabled: return

        # Restore memory of other patterns
        idx = self.iter % self.length
        for m, v in self.memory.items():
            setattr(self.dataset, m, v)

        # Set current pattern
        key = self.pattern[idx]
        setattr(self.dataset, key, self.pattern_cfg[key])

        # # Special care for n_rays, disabled perc loss & ssim loss
        # if key == 'n_rays':
        #     self.supervisor.perc_loss_weight = 0
        #     self.supervisor.ssim_loss_weight = 0
        # else:
        #     self.supervisor.perc_loss_weight = self.static.perc_loss_weight
        #     self.supervisor.ssim_loss_weight = self.static.ssim_loss_weight
