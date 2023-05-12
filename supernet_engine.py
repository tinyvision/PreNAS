import math
import sys
from typing import Iterable, Optional
from timm.utils.model import unwrap_model
import torch
import torch.distributed as dist
from copy import deepcopy
from timm.data import Mixup
from timm.utils import accuracy, ModelEma
from lib import utils
import random
import time
import json
from contextlib import ExitStack

def sample_a_cand(candidates, grouping=None, exclude=None):
    exclude = exclude or []
    while not grouping:
        idx = random.choice(range(len(candidates)))
        if idx not in exclude:
            return candidates[idx]
    while True:
        idx = random.choice(random.choice(grouping))
        if idx not in exclude:
            return candidates[idx]

def sample_candidates(candidates, eval=False, sandwich=0, sandwich_base=True, sandwich_top=True, shuffle=False, grouping=None):
    if eval:
        return candidates[0]
    else:
        if sandwich == 0:
            cand = sample_a_cand(candidates, grouping)
            if shuffle:
                cand = deepcopy(cand)
                random.shuffle(cand['mlp_ratio'])
                random.shuffle(cand['num_heads'])
            return cand
        else:
            base_cand = []
            top_cand = []
            exclude = []
            if sandwich_base:
                base_cand = [candidates[0]]
                exclude.append(0)
            if sandwich_top:
                top_cand = [candidates[-1]]
                exclude.append(len(candidates)-1)
            inter_cands = [sample_a_cand(candidates, grouping, exclude) for _ in range(sandwich)]
            return base_cand + inter_cands + top_cand

def sample_a_config(choices, efunc=random.choice):
    config = {}
    embed_dim = efunc(choices['embed_dim'])
    if isinstance(choices['depth'], dict):
        depth = efunc(choices['depth'][embed_dim])
    else:
        depth = efunc(choices['depth'])
    dimensions = ['mlp_ratio', 'num_heads']
    for dimension in dimensions:
        if isinstance(choices[dimension], dict):
            config[dimension] = [efunc(choices[dimension][embed_dim][i]) for i in range(depth)]
        else:
            config[dimension] = [efunc(choices[dimension]) for _ in range(depth)]
    config['embed_dim'] = [embed_dim] * depth
    config['layer_num'] = depth
    return config

def sample_configs(choices, eval=False, sandwich=0, sandwich_base=True, sandwich_top=True):
    if eval:
        return sample_a_config(choices, min)
    else:
        if sandwich == 0:
            return sample_a_config(choices)
        else:
            base_config = [sample_a_config(choices, min)] if sandwich_base else []
            top_config = [sample_a_config(choices, max)] if sandwich_top else []
            inter_configs = [sample_a_config(choices) for _ in range(sandwich)]
            return base_config + inter_configs + top_config

def bp_once(loss, loss_scaler=None, create_graph=False):
    if loss_scaler:
        loss_scaler._scaler.scale(loss).backward(create_graph=create_graph)
    else:
        loss.backward()

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    amp: bool = True, teacher_model: torch.nn.Module = None,
                    teach_loss: torch.nn.Module = None, choices=None, mode='super', retrain_config=None,
                    print2file=False, candidates=None, sandwich=0, sandwich_base=True, sandwich_top=True,
                    shuffle=False, grouping=None):
    model.train()
    criterion.train()

    # set random seed
    random.seed(epoch)

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    if mode == 'retrain':
        config = retrain_config
        model_module = unwrap_model(model)
        print("DEBUG:retrain {}".format(config), force=print2file)
        model_module.set_sample_config(config=config)
        print("DEBUG:retrain {}".format(model_module.get_sampled_params_numel(config)), force=print2file)

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # sample random config
        if mode == 'super':
            sandwich_args = {'sandwich': sandwich,
                             'sandwich_base': sandwich_base,
                             'sandwich_top': sandwich_top,
                            }
            if candidates is not None:
                config = sample_candidates(candidates, **sandwich_args, shuffle=shuffle, grouping=grouping)
            else:
                config = sample_configs(choices, **sandwich_args)
            if isinstance(config, dict):
                config = [config]
            model_module = unwrap_model(model)
            #model_module.set_sample_config(config=config)
        elif mode == 'retrain':
            config = retrain_config
            model_module = unwrap_model(model)
            model_module.set_sample_config(config=config)
        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        optimizer.zero_grad()
        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order

        loss_value = 0.0
        with ExitStack() if not amp else torch.cuda.amp.autocast():
            if teacher_model:
                with torch.no_grad():
                    teach_output = teacher_model(samples)
                _, teacher_label = teach_output.topk(1, 1, True, True)
                del teach_output
                teacher_label.squeeze_()
                factor = 1.0 / len(config)
                for cfg in config:
                    model_module.set_sample_config(cfg)
                    outputs = model(samples)
                    # gt
                    loss = 0.5 * factor * criterion(outputs, targets)
                    bp_once(loss, loss_scaler, is_second_order)
                    loss_value += loss.item()
                    # teacher
                    loss = 0.5 * factor * teach_loss(outputs, teacher_label)
                    bp_once(loss, loss_scaler, is_second_order)
                    loss_value += loss.item()
            else:
                factor = 1.0 / len(config)
                for cfg in config:
                    model_module.set_sample_config(cfg)
                    loss = factor * criterion(model(samples), targets)
                    bp_once(loss, loss_scaler, is_second_order)
                    loss_value += loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            #sys.exit(1)
            continue

        if amp:
            loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)
        else:
            optimizer.step()

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # only check at the end of epoch (avoid flooding)
    print("DEBUG:train {}".format(config), force=print2file)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(data_loader, model, device, amp=True, choices=None, mode='super', retrain_config=None, print2file=False, candidates=None, eval_crops=1):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    if mode == 'super':
        if candidates is not None:
            config = sample_candidates(candidates, eval=True)
        else:
            config = sample_configs(choices, eval=True)
        config = [config]
        if utils.is_dist_avail_and_initialized():
            dist.broadcast_object_list(config, src=0)
        config = config[0]
        model_module = unwrap_model(model)
        model_module.set_sample_config(config=config)
    else:
        config = retrain_config
        model_module = unwrap_model(model)
        model_module.set_sample_config(config=config)


    print("DEBUG:eval sampled model config: {}".format(config), force=print2file)
    parameters = model_module.get_sampled_params_numel(config)
    print("DEBUG:eval sampled model parameters: {}".format(parameters), force=print2file)

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        if eval_crops > 1:
            bs, ncrops, c, h, w = images.size()
            images = images.view(-1, c, h, w)

        # compute output
        with ExitStack() if not amp else torch.cuda.amp.autocast():
            output = model(images)
            if eval_crops > 1:
                output = output.view(bs, ncrops, -1).mean(1)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    metric_logger.update(n_parameters=parameters)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
