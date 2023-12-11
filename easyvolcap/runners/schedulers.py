import numpy as np
from torch.optim.optimizer import Optimizer
from easyvolcap.engine import SCHEDULERS
from easyvolcap.utils.base_utils import dotdict
from torch.optim.lr_scheduler import _LRScheduler


@SCHEDULERS.register_module()
class MultiLR(_LRScheduler):
    def __init__(self, optimizer, decay_iter, scheduler_cfgs, last_epoch=-1, verbose=False):
        self.schedulers = dotdict()
        self.names = [param_group['name'] for param_group in optimizer.param_groups]
        # values = self._get_optimizer_lr(optimizer)
        for name, scheduler_cfg in scheduler_cfgs.items():
            scheduler = SCHEDULERS.build(scheduler_cfg, optimizer=optimizer, decay_iter=decay_iter)
            self.schedulers[name] = scheduler
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        result = []
        for name, sched in self.schedulers.items():
            idx = self.names.index(name)
            result.append(sched.get_last_lr()[idx])
        return result

    @staticmethod
    def _set_optimizer_lr(optimizer, values):
        for param_group in optimizer.param_groups:
            param_group['lr'] = values[param_group['name']]

    @staticmethod
    def _get_optimizer_lr(optimizer):
        values = dotdict()
        for param_group in optimizer.param_groups:
            values[param_group['name']] = param_group['lr']
        return values

    def step(self, epoch=None):
        if self.last_epoch != -1:
            values = self._get_optimizer_lr(self.optimizer)
            for name, sched in self.schedulers.items():
                sched.step()
                values[name] = self._get_optimizer_lr(self.optimizer)[name]
                self._set_optimizer_lr(self.optimizer, values)
        super().step()


@SCHEDULERS.register_module()
class NoopLR(_LRScheduler):
    def __init__(self, optimizer, last_epoch=-1, **kwargs):
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        lrs = [base_lr for base_lr in self.base_lrs]
        return lrs


@SCHEDULERS.register_module()
class ExponentialLR(_LRScheduler):
    def __init__(self,
                 optimizer,  # object
                 decay_iter,  # no default for this config
                 gamma=0.1,
                 min_lr=5e-5,
                 last_epoch=-1):
        self.decay_iter = decay_iter
        self.gamma = gamma
        self.min_lr = min_lr
        super(ExponentialLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        lrs = [base_lr * self.gamma ** (self.last_epoch / self.decay_iter) for base_lr in self.base_lrs]
        lrs = [max(lr, self.min_lr) for lr in lrs]
        return lrs


@SCHEDULERS.register_module()
class WarmupExponentialLR(_LRScheduler):
    def __init__(self,
                 optimizer,  # object
                 decay_iter,
                 warmup_factor=1.0 / 3,
                 warmup_epochs=1,
                 warmup_method="linear",
                 gamma=0.1,
                 min_lr=5e-5,
                 last_epoch=-1):
        self.warmup_factor = warmup_factor
        self.warmup_epochs = warmup_epochs
        self.warmup_method = warmup_method
        self.decay_iter = decay_iter
        self.gamma = gamma
        self.min_lr = min_lr
        super(WarmupExponentialLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_epochs:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = float(self.last_epoch) / self.warmup_epochs
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        lrs = [base_lr * warmup_factor * self.gamma ** (self.last_epoch / self.decay_iter) for base_lr in self.base_lrs]
        lrs = [max(lr, self.min_lr) for lr in lrs]
        return lrs


@SCHEDULERS.register_module()
class NeuSScheduler(_LRScheduler):
    def __init__(self,
                 optimizer,
                 decay_iter,  # no default for this config
                 warm_up_end=500,
                 learning_rate_alpha=0.05,
                 last_epoch=-1):
        self.warm_up_end = warm_up_end
        self.learning_rate_alpha = learning_rate_alpha
        self.decay_iter = decay_iter
        super(NeuSScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        learning_factor = 1.0
        if self.last_epoch < self.warm_up_end:
            learning_factor = self.last_epoch / self.warm_up_end
        else:
            alpha = self.learning_rate_alpha
            progress = (self.last_epoch - self.warm_up_end) / (self.decay_iter - self.warm_up_end)
            learning_factor = (np.cos(np.pi * progress) + 1.0) * 0.5 * (1 - alpha) + alpha
        lrs = [base_lr * learning_factor for base_lr in self.base_lrs]
        return lrs


@SCHEDULERS.register_module()
class MultiStepWarmupScheduler(_LRScheduler):
    def __init__(self,
                 optimizer,
                 warm_up_end=5000,
                 milestones=[300000, 400000],
                 gamma=0.1,
                 last_epoch=-1):
        self.warm_up_end = warm_up_end
        self.milestones = milestones
        self.gamma = gamma
        super(MultiStepWarmupScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self) -> float:
        learning_factor = 1.0
        if self.last_epoch < self.warm_up_end:
            learning_factor = self.last_epoch / self.warm_up_end
        else:
            index = np.searchsorted(self.milestones, self.last_epoch, side='left')
            learning_factor = self.gamma ** index
        lrs = [base_lr * learning_factor for base_lr in self.base_lrs]
        return lrs
