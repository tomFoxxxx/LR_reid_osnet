import torch
import math
from torch.optim import lr_scheduler

__all__ = ['init_lr_schedule','show_lr_schedule']

def init_lr_schedule(schedule, warm_up_epoch, half_cos_period, lr_milestone, gamma, stepsize):
    warm_up_epochs = warm_up_epoch
    half_cos_periods = half_cos_period
    lr_milestones = lr_milestone
    gammas = gamma
    stepsizes = stepsize
    if schedule == 'multistep_lr':
        # MultiStepLR without warm up
        multistep_lr = lambda epoch: gammas ** len([m for m in lr_milestones if m <= epoch])
        print('using multistep_lr')
        return multistep_lr
    elif schedule == 'warm_up_with_multistep_lr':
        # warm_up_with_multistep_lr
        warm_up_with_multistep_lr = lambda epoch: (epoch) / warm_up_epochs if epoch < warm_up_epochs \
            else gammas ** len([m for m in lr_milestones if m <= epoch])
        print('using warm_up_with_multistep_lr')
        return warm_up_with_multistep_lr
    elif schedule == 'step_lr':
        # step_lr
        step_lr = lambda epoch: gammas ** ((epoch - warm_up_epochs) // stepsizes)
        print('using step_lr')
        return step_lr
    elif schedule == 'warm_up_with_step_lr':
        # warm_up_with_step_lr
        warm_up_with_step_lr = lambda epoch: (epoch) / warm_up_epochs if epoch < warm_up_epochs \
            else gammas ** ((epoch - warm_up_epochs) // stepsizes)
        print('using warm_up_with_step_lr')
        return warm_up_with_step_lr
    elif schedule == 'warm_up_with_cosine_lr':
        warm_up_with_cosine_lr = lambda epoch: (epoch) / warm_up_epochs if epoch < warm_up_epochs \
            else 0.5 * (math.cos((epoch - warm_up_epochs) / (half_cos_periods - warm_up_epochs) * math.pi) + 1)
        print('using warm_up_with_cosine_lr')
        return warm_up_with_cosine_lr
    else:
        raise KeyError("Unsupported lr_schedule: {}".format(schedule))


def show_lr_schedule():
    return ['multistep_lr','warm_up_with_multistep_lr','step_lr','warm_up_with_step_lr','warm_up_with_cosine_lr']

