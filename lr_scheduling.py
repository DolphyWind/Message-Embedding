import math
from typing import List
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


class LinearWarmupCosineDecay(LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        start_factor: float,
        end_factor: float,
        warmup_steps: int,
        total_steps: int,
        last_epoch: int = -1,
    ) -> None:
        assert warmup_steps >= 0
        assert total_steps > warmup_steps
        assert start_factor >= 0.0
        assert end_factor >= 0.0

        self.start_factor: float = start_factor
        self.end_factor: float = end_factor
        self.warmup_steps: int = warmup_steps
        self.total_steps: int = total_steps
        self.decay_steps: int = total_steps - warmup_steps

        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        step: int = self.last_epoch + 1
        lrs: List[float] = []

        for base_lr in self.base_lrs:
            start_lr: float = base_lr * self.start_factor
            end_lr: float = base_lr * self.end_factor

            if step < self.warmup_steps:
                if self.warmup_steps == 0:
                    lr = base_lr
                else:
                    lr = start_lr + (base_lr - start_lr) * (step / self.warmup_steps)
            else:
                decay_step: int = min(step - self.warmup_steps, self.decay_steps)
                cosine: float = 0.5 * (1.0 + math.cos(math.pi * decay_step / self.decay_steps))
                lr = end_lr + (base_lr - end_lr) * cosine

            lrs.append(lr)

        return lrs
