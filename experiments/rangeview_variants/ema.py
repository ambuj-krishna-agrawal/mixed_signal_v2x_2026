"""Exponential moving average of model weights.

Why this matters for us specifically: the LiDAR-MOS schedule deliberately holds
the learning rate near its peak (99% at epoch 0, still 82% at epoch 19) so the
model keeps exploring rather than settling into a memorised minimum. The cost is
that the weights are always in motion, so held-out metrics sampled at any single
step bounce -- our mid-epoch val loss swings ~25% while the epoch-end full-val
number falls smoothly.

An EMA converts that motion from a nuisance into an averaging benefit: it tracks
a smoothed trajectory through the region the optimiser is exploring, which is
usually a better model than any individual point in it. It also makes checkpoint
selection far less of a lottery -- our 80-frame subset metric has already named
the wrong checkpoint once.
"""
import copy
import torch


class ModelEMA:
    def __init__(self, model, decay=0.999, warmup=2000):
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay = decay
        self.warmup = warmup
        self.updates = 0

    def _d(self):
        # ramp the decay in: a cold EMA initialised from random weights would
        # otherwise lag far behind for the whole first epoch
        if self.warmup <= 0:
            return self.decay
        return self.decay * (1 - torch.exp(torch.tensor(-self.updates / self.warmup)).item())

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        d = self._d()
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(d).add_(msd[k].detach(), alpha=1 - d)
            else:
                v.copy_(msd[k])          # integer buffers (e.g. num_batches_tracked)
