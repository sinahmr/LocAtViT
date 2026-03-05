from typing import List

import torch
from timm.scheduler.scheduler import Scheduler


class ConstCooldownScheduler(Scheduler):
    def __init__(
            self,
            optimizer: torch.optim.Optimizer,
            num_epochs=300,
    ) -> None:
        super().__init__(
            optimizer,
            param_group_field="lr",
        )
        self.num_epochs = num_epochs
        self.cooldown_start = 0.8 * num_epochs

    def _get_lr(self, t: int) -> List[float]:
        if t < self.cooldown_start:
            return self.base_values
        if t >= self.num_epochs:
            t = self.num_epochs - 1
        rate = (self.num_epochs - t) / (self.num_epochs - self.cooldown_start)
        return [rate * v for v in self.base_values]


class OneCycleScheduler(Scheduler):
    def __init__(
            self,
            optimizer: torch.optim.Optimizer,
            num_epochs=300,
            critical_epoch=100,
            start_lr=1e-7,
            end_lr=1e-7,
            t_in_epochs=False,
    ) -> None:
        super().__init__(
            optimizer,
            param_group_field='lr',
            t_in_epochs=t_in_epochs,
        )
        self.num_epochs = num_epochs
        self.critical_epoch = critical_epoch
        self.start_lr = start_lr
        self.end_lr = end_lr
        super().update_groups(start_lr)

    def _get_lr(self, t: int) -> List[float]:
        if t < self.critical_epoch:
            return [self.start_lr + t * (v - self.start_lr) / self.critical_epoch for v in self.base_values]
        elif self.critical_epoch <= t < self.num_epochs:
            return [v - (t - self.critical_epoch) * (v - self.end_lr) / (self.num_epochs - self.critical_epoch) for v in self.base_values]
        else:
            return [self.end_lr for _ in self.base_values]
