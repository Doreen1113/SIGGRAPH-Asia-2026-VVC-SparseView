# FreeTimeGS++
# 2025-2026 Lucas Yunkyu Lee <lucaslee@postech.ac.kr>, SNU VGI Lab

from typing import Mapping

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

class OptimizerCollection(Mapping[str, Optimizer]):
    _optimizers: dict[str, Optimizer]

    def __init__(self, optimizers: Mapping[str, Optimizer]):
        self._optimizers = dict(optimizers)

    def step(self, *args, **kwargs):
        for opt in self._optimizers.values():
            opt.step(*args, **kwargs)

    def zero_grad(self, set_to_none: bool = True):
        for opt in self._optimizers.values():
            opt.zero_grad(set_to_none)

    def __getitem__(self, key: str) -> Optimizer:
        return self._optimizers[key]

    def __iter__(self):
        return iter(self._optimizers)

    def __len__(self) -> int:
        return len(self._optimizers)

    def asdict(self) -> dict[str, Optimizer]:
        return self._optimizers


class LRSchedulerCollection(dict[str, LRScheduler]):
    def step(self):
        for lr_scheduler in self.values():
            lr_scheduler.step()
