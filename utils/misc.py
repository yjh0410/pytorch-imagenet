import torch
import torch.nn as nn
import math
from thop import profile
from copy import deepcopy


def is_parallel(model):
    # Returns True if model is of type DP or DDP
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)


def FLOPs_and_Params(model, size, device):
    x = torch.randn(1, 3, size, size).to(device)

    flops, params = profile(model, inputs=(x, ))
    print('FLOPs : ', flops / 1e9, ' G')
    print('Params : ', params / 1e6, ' M')


if __name__ == "__main__":
    pass



# Exponential Moving Average for model
class ModelEMA(object):
    def __init__(self, model, decay=0.9999, updates=0):
        # create EMA
        self.ema = deepcopy(model.module if is_parallel(model) else model).eval()  # FP32 EMA
        self.updates = updates
        self.decay = lambda x: decay * (1 - math.exp(-x / 2000.))
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        # Update EMA parameters
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)

            msd = model.module.state_dict() if is_parallel(model) else model.state_dict()  # model state_dict
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1. - d) * msd[k].detach()
