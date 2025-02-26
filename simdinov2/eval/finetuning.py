# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import datetime
import io
import json
import logging
import math
import os
import sys
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torchvision.transforms as transforms
from numpy import inf
from PIL import Image
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data.mixup import Mixup
from timm.data.random_erasing import RandomErasing
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import ModelEma, accuracy
from torch import optim

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..')))
from simdinov2.data import make_dataset
from simdinov2.data.transforms import make_finetuning_transform
from simdinov2.eval.metrics import MetricType
from simdinov2.eval.setup import get_args_parser as get_setup_args_parser
from simdinov2.eval.setup import setup_and_build_model
from simdinov2.eval.rand_aug import rand_augment_transform
logger = logging.getLogger("dino")


class ModelWithClassifier(nn.Module):

    def __init__(self, feature_model, embed_dim, num_classes=1000, use_multi_stage_feat=False, use_cls=False):
        super().__init__()
        self.feature_model = feature_model
        self.feature_model.use_mean_pooling = not use_cls
        self.head = nn.Linear(embed_dim, num_classes)
        self.use_multi_stage_feat = use_multi_stage_feat
        self.use_cls = use_cls
        if self.use_cls:
            logger.info("Using features: x_norm_clstoken")
        else:
            logger.info("Using features: x_mean_pooling")

    def forward(self, images):
        if self.use_cls:
            features = self.feature_model.forward_features(images)["x_norm_clstoken"]
        else:
            if self.use_multi_stage_feat:
                features = self.feature_model.forward_multistage_features(images)["x_mean_pooling"]
            else:
                features = self.feature_model.forward_features(images)["x_mean_pooling"]
        logit = self.head(features)
        return logit

    def get_depths(self):
        return self.feature_model.get_depths()

    def no_weight_decay(self):
        names = self.feature_model.no_weight_decay()
        names = ['feature_model.' + name for name in names]
        return names


def get_parameter_groups(
        model, weight_decay=1e-5, skip_list=(), get_num_layer=None, get_layer_scale=None
    ):
    parameter_group_names = {}
    parameter_group_vars = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            group_name = "no_decay"
            this_weight_decay = 0.
        else:
            group_name = "decay"
            this_weight_decay = weight_decay
        if get_num_layer is not None:
            layer_id = get_num_layer(name)
            group_name = "layer_%d_%s" % (layer_id, group_name)
        else:
            layer_id = None

        if group_name not in parameter_group_names:
            if get_layer_scale is not None:
                scale = get_layer_scale(layer_id)
            else:
                scale = 1.

            parameter_group_names[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": scale
            }
            parameter_group_vars[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": scale
            }

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)
    print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
    return list(parameter_group_vars.values())


def create_optimizer(
        args, model, get_num_layer=None, get_layer_scale=None,
        filter_bias_and_bn=True, skip_list=None
    ):
    opt_lower = args.opt.lower()
    weight_decay = args.weight_decay
    if weight_decay and filter_bias_and_bn:
        skip = {}
        if skip_list is not None:
            skip = skip_list
        elif hasattr(model, 'no_weight_decay'):
            skip = model.no_weight_decay()
        parameters = get_parameter_groups(model, weight_decay, skip, get_num_layer, get_layer_scale)
        weight_decay = 0.
    else:
        parameters = model.parameters()

    opt_args = dict(lr=args.learning_rate, weight_decay=weight_decay)
    opt_args['eps'] = 1e-8

    opt_split = opt_lower.split('_')
    opt_lower = opt_split[-1]
    if opt_lower == 'sgd' or opt_lower == 'nesterov':
        opt_args.pop('eps', None)
        optimizer = optim.SGD(parameters, momentum=args.momentum, nesterov=True, **opt_args)
    elif opt_lower == 'momentum':
        opt_args.pop('eps', None)
        optimizer = optim.SGD(parameters, momentum=args.momentum, nesterov=False, **opt_args)
    elif opt_lower == 'adam':
        optimizer = optim.Adam(parameters, **opt_args)
    elif opt_lower == 'adamw':
        optimizer = optim.AdamW(parameters, **opt_args)
    else:
        assert False and "Invalid optimizer"
        raise ValueError

    return optimizer


class LayerDecayValueAssigner(object):

    def __init__(self, values, prefix, net_type, actived_block_idx, depths=None):
        assert net_type in ['swin', 'vit', 'resnet']
        self.values = values
        self.depths = depths
        self.prefix = prefix
        self.net_type = net_type
        if net_type == 'resnet':
            assert isinstance(actived_block_idx, list)
            assert len(actived_block_idx) == 2
            self.block_id_map = []
            for actived_block_idx_i in actived_block_idx:
                self.block_id_map.append({str(x): i for i, x in enumerate(actived_block_idx_i)})
        else:
            self.block_id_map = {str(x): i for i, x in enumerate(actived_block_idx)}

    def get_scale(self, layer_id):
        return self.values[layer_id]

    def get_num_layer_for_resnet(self, var_name, num_max_layer, depths):
        if var_name == f"{self.prefix}.mask_token":
            return 0
        elif var_name.startswith(f"{self.prefix}.conv1"):
            return 0
        elif var_name.startswith(f"{self.prefix}.norm1"):
            return 0
        elif var_name.startswith(f"{self.prefix}.layer"):
            stage_id = int(var_name.split('.')[1].replace('layer', '')) - 1
            if stage_id == 1:
                block_id = self.block_id_map[0].get(var_name.split('.')[3], -1)
            elif stage_id == 2:  
                block_id = self.block_id_map[1].get(var_name.split('.')[3], -1)
            else:
                block_id = int(var_name.split('.')[3])
            layer_id = sum(depths[:stage_id]) + block_id
            if block_id != -1:
                print(f'resnet-{stage_id}-{layer_id}', var_name)
            else:
                return 0 # not activated parameters
            return layer_id + 1
        else:
            return num_max_layer - 1

    def get_num_layer_for_swin(self, var_name, num_max_layer, depths):
        if var_name in (
            f"{self.prefix}.mask_token", f"{self.prefix}.pos_embed"
        ):
            return 0
        elif var_name.startswith(f"{self.prefix}.patch_embed"):
            return 0
        elif var_name.startswith(f"{self.prefix}.stages"):
            stage_id = int(var_name.split('.')[2])
            if stage_id == 2:
                if 'blocks' in var_name:
                    block_id = self.block_id_map.get(var_name.split('.')[4], -1)
                    if block_id != -1:
                        self.cur_block_id = block_id
                else:
                    block_id = self.cur_block_id
            else:
                if 'blocks' in var_name: 
                    block_id = int(var_name.split('.')[4])
                    self.cur_block_id = block_id
                else: 
                    block_id = self.cur_block_id
            layer_id = sum(depths[:stage_id]) + block_id
            if block_id != -1:
                print(f'swin-{layer_id}', var_name)
            else:
                return 0 # not activated parameters
            return layer_id + 1
        else:
            return num_max_layer - 1

    def get_num_layer_for_vit(self, var_name, num_max_layer):
        if var_name in (
            f"{self.prefix}.cls_token", f"{self.prefix}.mask_token", f"{self.prefix}.pos_embed"
        ):
            return 0
        elif var_name.startswith(f"{self.prefix}.patch_embed"):
            return 0
        elif var_name.startswith(f"{self.prefix}.blocks"):
            layer_id = self.block_id_map.get(var_name.split('.')[2], -1)
            return layer_id + 1
        else:
            return num_max_layer - 1

    def get_layer_id(self, var_name):
        if self.net_type == 'swin':
            return self.get_num_layer_for_swin(var_name, len(self.values), self.depths) 
        if self.net_type == 'resnet':
            return self.get_num_layer_for_resnet(var_name, len(self.values), self.depths) 
        if self.net_type == 'vit':
            return self.get_num_layer_for_vit(var_name, len(self.values))   


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0,
                     start_warmup_value=0, warmup_steps=-1):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_steps > 0:
        warmup_iters = warmup_steps
    print("Set warmup steps = %d" % warmup_iters)
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = np.array(
        [final_value + 0.5 * (base_value - final_value) * (1 + math.cos(math.pi * i / (len(iters)))) for i in iters])

    schedule = np.concatenate((warmup_schedule, schedule))

    assert len(schedule) == epochs * niter_per_ep
    return schedule


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = self.get_grad_norm_(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)

    def get_grad_norm_(self, parameters, norm_type: float = 2.0) -> torch.Tensor:
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]
        norm_type = float(norm_type)
        if len(parameters) == 0:
            return torch.tensor(0.)
        device = parameters[0].grad.device
        if norm_type == inf:
            total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
        else:
            total_norm = torch.norm(torch.stack(
                [torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
        return total_norm


class MetricLogger2(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


def get_logger(file_path_name):
    logger = logging.getLogger()
    logger.setLevel('INFO')
    BASIC_FORMAT = "%(levelname)s:%(message)s"
    DATE_FORMAT = ''
    formatter = logging.Formatter(BASIC_FORMAT, DATE_FORMAT)
    chlr = logging.StreamHandler()
    chlr.setFormatter(formatter)
    chlr.setLevel('INFO')
    fhlr = logging.FileHandler(file_path_name)
    fhlr.setFormatter(formatter)
    logger.addHandler(chlr)
    logger.addHandler(fhlr)
    return logger


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def _pil_interp(method):
    if method == 'bicubic':
        return Image.BICUBIC
    elif method == 'lanczos':
        return Image.LANCZOS
    elif method == 'hamming':
        return Image.HAMMING
    else:
        # default bilinear, do we want to allow nearest?
        return Image.BILINEAR


def strong_transforms(
    img_size=224,
    scale=(0.08, 1.0),
    ratio=(0.75, 1.3333333333333333),
    hflip=0.5,
    vflip=0.0,
    color_jitter=0.4,
    auto_augment="rand-m9-mstd0.5-inc1",
    interpolation="random",
    use_prefetcher=True,
    mean=IMAGENET_DEFAULT_MEAN,  # (0.485, 0.456, 0.406)
    std=IMAGENET_DEFAULT_STD,  # (0.229, 0.224, 0.225)
    re_prob=0.25,
    re_mode="pixel",
    re_count=1,
    re_num_splits=0,
    color_aug=False,
    strong_ratio=0.45,
):
    """
    for use in a mixing dataset that passes
     * all data through the first (primary) transform, called the 'clean' data
     * a portion of the data through the secondary transform
     * normalizes and converts the branches above with the third, final transform
    """

    scale = tuple(scale or (0.08, 1.0))  # default imagenet scale range
    ratio = tuple(ratio or (3.0 / 4.0, 4.0 / 3.0))  # default imagenet ratio range

    primary_tfl = []
    if hflip > 0.0:
        primary_tfl += [transforms.RandomHorizontalFlip(p=hflip)]
    if vflip > 0.0:
        primary_tfl += [transforms.RandomVerticalFlip(p=vflip)]

    secondary_tfl = []
    if auto_augment:
        assert isinstance(auto_augment, str)
        if isinstance(img_size, tuple):
            img_size_min = min(img_size)
        else:
            img_size_min = img_size
        aa_params = dict(
            translate_const=int(img_size_min * strong_ratio),
            img_mean=tuple([min(255, round(255 * x)) for x in mean]),
        )
        if interpolation and interpolation != "random":
            aa_params["interpolation"] = _pil_interp(interpolation)
        if auto_augment.startswith("rand"):
            secondary_tfl += [rand_augment_transform(auto_augment, aa_params)]
    if color_jitter is not None and color_aug:
        # color jitter is enabled when not using AA
        flip_and_color_jitter = [
            transforms.RandomApply(
                [
                    transforms.ColorJitter(
                        brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
                    )
                ],
                p=0.8,
            ),
            transforms.RandomGrayscale(p=0.2),
        ]
        secondary_tfl += flip_and_color_jitter

    if interpolation == "random":
        interpolation = (Image.BILINEAR, Image.BICUBIC)
    else:
        interpolation = _pil_interp(interpolation)
    final_tfl = [
        transforms.RandomResizedCrop(
            size=img_size, scale=scale, ratio=ratio, interpolation=Image.BICUBIC
        )
    ]
    if use_prefetcher:
        # prefetcher and collate will handle tensor conversion and norm
        final_tfl += [transforms.ToTensor()]
    else:
        final_tfl += [
            transforms.ToTensor(),
            transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std)),
        ]
    if re_prob > 0.0:
        final_tfl.append(
            RandomErasing(
                re_prob,
                mode=re_mode,
                max_count=re_count,
                num_splits=re_num_splits,
                device="cpu",
            )
        )
    return transforms.Compose(primary_tfl + secondary_tfl + final_tfl)

def get_args_parser(
    description: Optional[str] = None,
    parents: Optional[List[argparse.ArgumentParser]] = [],
    add_help: bool = True,
):
    setup_args_parser = get_setup_args_parser(parents=parents, add_help=False)
    parents = [setup_args_parser]
    parser = argparse.ArgumentParser(
        description=description,
        parents=parents,
        add_help=add_help,
    )
    parser.add_argument(
        "--arch-name",
        type=str,
        default="vit",
        help="Architecture name: swin, vit, or resnet",
    )
    parser.add_argument(
        "--net-type",
        type=str,
        default="base",
        help="Network type for vit or swin, for example: samll, base, large",
    )
    parser.add_argument(
        "--train-dataset",
        dest="train_dataset_str",
        type=str,
        help="Training dataset",
    )
    parser.add_argument(
        "--val-dataset",
        dest="val_dataset_str",
        type=str,
        help="Validation dataset",
    )
    parser.add_argument(
        "--test-datasets",
        dest="test_dataset_strs",
        type=str,
        nargs="+",
        help="Test datasets, none to reuse the validation dataset",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Batch Size (per GPU)",
    )
    parser.add_argument(
        '--input-size',
        type=int,
        help='images input size')
    parser.add_argument(
        "--num-workers",
        type=int,
        help="Number de Workers",
    )
    parser.add_argument(
        "--epoch-length",
        type=int,
        help="Length of an epoch in number of iterations",
    )
    parser.add_argument(
        "--save-checkpoint-frequency",
        type=int,
        help="Number of epochs between two named checkpoint saves.",
    )
    parser.add_argument(
        "--eval-period-iterations",
        type=int,
        help="Number of iterations between two evaluations.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        help="Learning rate for finetuning.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Whether to not resume from existing checkpoints",
    )
    parser.add_argument(
        "--val-metric-type",
        type=MetricType,
        choices=list(MetricType),
        help="Validation metric",
    )
    parser.add_argument(
        "--test-metric-types",
        type=MetricType,
        choices=list(MetricType),
        nargs="+",
        help="Evaluation metric",
    )
    parser.add_argument(
        "--classifier-fpath",
        type=str,
        help="Path to a file containing pretrained linear classifiers",
    )
    parser.add_argument(
        "--val-class-mapping-fpath",
        type=str,
        help="Path to a file containing a mapping to adjust classifier outputs",
    )
    parser.add_argument(
        "--test-class-mapping-fpaths",
        nargs="+",
        type=str,
        help="Path to a file containing a mapping to adjust classifier outputs",
    )
    parser.add_argument(
        '--color-jitter',
        type=float,
        help='Color jitter factor (default: 0.4)'
    )
    parser.add_argument(
        '--aa', type=str, default='rand-m9-mstd0.5-inc1',
        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'
    )
    parser.add_argument(
        '--train-interpolation',
        type=str,
        help='Training interpolation (random, bilinear, bicubic default: "bicubic")'
    )
    parser.add_argument(
        '--reprob',
        type=float,
        help='Random erase prob (default: 0.25)'
    )
    parser.add_argument(
        '--opt',
        type=str,
        help='Optimizer (default: "adamw")'
    )
    parser.add_argument(
        '--weight-decay',
        type=float,
        help='Weight decay (default: 0.05)'
    )
    parser.add_argument(
        '--momentum',
        type=float,
        help='SGD momentum (default: 0.9)'
    )
    parser.add_argument(
        '--min-lr',
        type=float,
        help='lower lr bound for cyclic schedulers that hit 0 (1e-6)'
    )
    parser.add_argument(
        '--warmup-epochs',
        type=int,
        help='epochs to warmup LR, if scheduler supports'
    )
    parser.add_argument(
        '--warmup-steps',
        type=int, default=-1,
        help='num of steps to warmup LR, will overload warmup_epochs if set > 0'
    )
    parser.add_argument(
        '--layer-decay',
        type=float,
        help='layer lr decay rate (default: 0.9)'
   )
    # * Mixup params
    parser.add_argument(
        '--mixup',
        type=float,
        help='mixup alpha, mixup enabled if > 0.'
    )
    parser.add_argument(
        '--cutmix',
        type=float,
        help='cutmix alpha, cutmix enabled if > 0.'
    )
    parser.add_argument(
        '--cutmix-minmax',
        type=float, nargs='+', default=None,
        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)'
    )
    parser.add_argument(
        '--mixup-prob',
        type=float, default=1.0,
        help='Probability of performing mixup or cutmix when either/both is enabled'
    )
    parser.add_argument(
        '--mixup-switch-prob',
        type=float, default=0.5,
        help='Probability of switching to cutmix when both mixup and cutmix enabled'
    )
    parser.add_argument(
        '--mixup-mode',
        type=str, default='batch',
        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"'
    )
    parser.add_argument(
        '--model-ema', action='store_true', 
        help='Using model EMA in training',
    )
    parser.add_argument(
        '--model-ema-decay', type=float, 
    )
    parser.add_argument(
        '--smoothing',
        type=float,
        help='Label smoothing (default: 0.1)'
    )
    parser.add_argument(
        '--pin_mem',
        action='store_true',
        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.'
    )
    parser.add_argument(
        '--clip_grad',
        type=float, default=None,
         help='Clip gradient norm (default: None, no clipping)'
    )
    parser.add_argument(
        '--drop_path', 
        type=float, default=0.1, 
        help='Drop path rate (default: 0.1)'
    )
    parser.set_defaults(
        train_dataset_str="ImageNet:split=TRAIN",
        val_dataset_str="ImageNet:split=VAL",
        test_dataset_strs=None,
        input_size=224,
        epochs=200,
        warmup_epochs=20,
        batch_size=128,
        num_workers=32,
        pin_mem=True,
        epoch_length=1250,
        color_jitter=0.4,
        reprob=0.25,
        train_interpolation="bicubic",
        save_checkpoint_frequency=20,
        eval_period_iterations=1250,
        opt='adamw',
        weight_decay=0.05,
        momentum=0.9,
        learning_rate=0.0012,
        min_lr=1e-6,
        layer_decay=0.9,
        mixup=0.8,
        cutmix=1.0,
        smoothing=0.1,
        model_ema=True,
        model_ema_decay=0.7,
        val_metric_type=MetricType.MEAN_ACCURACY,
        test_metric_types=None,
        classifier_fpath=None,
        val_class_mapping_fpath=None,
        test_class_mapping_fpaths=[None],
    )
    return parser

def train_class_batch(model, samples, target, criterion):
    outputs = model(samples)
    loss = criterion(outputs, target)
    return loss, outputs


@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = MetricLogger2(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[-1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
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

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train_one_epoch(
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        data_loader: Iterable,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        epoch: int,
        loss_scaler,
        max_norm: float = 0,
        mixup_fn: Optional[Mixup] = None,
        start_steps=None,
        lr_schedule_values=None,
        wd_schedule_values=None,
        num_training_steps_per_epoch=None,
        update_freq=None,
        model_ema: Optional[ModelEma] = None,
    ):
    model.train(True)
    metric_logger = MetricLogger2(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    if loss_scaler is None:
        model.zero_grad()
        model.micro_steps = 0
    else:
        optimizer.zero_grad()
    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        step = data_iter_step // update_freq
        if step >= num_training_steps_per_epoch:
            continue
        it = start_steps + step  # global training iteration
        # Update LR & WD for the first acc
        if lr_schedule_values is not None or wd_schedule_values is not None and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        if loss_scaler is None:
            samples = samples.half()
            loss, output = train_class_batch(
                model, samples, targets, criterion)
        else:
            with torch.cuda.amp.autocast():
                loss, output = train_class_batch(
                    model, samples, targets, criterion)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            logger.info("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)
        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss /= update_freq
        grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                parameters=model.parameters(), create_graph=is_second_order,
                                update_grad=(data_iter_step + 1) % update_freq == 0)
        if (data_iter_step + 1) % update_freq == 0:
            optimizer.zero_grad()
            if model_ema is not None:
                model_ema.update(model)
        loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        if mixup_fn is None:
            class_acc = (output.max(-1)[-1] == targets).float().mean()
            metric_logger.update(class_acc=class_acc)
        else:
            class_acc = None
        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_scale=loss_scale_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



def run_finetnuing(args):

    if not Path(args.pretrained_weights).exists():
        raise Exception(f'Pretrained model not found: {args.pretrained_weights}')

    # training setting 
    assert args.arch_name in ['vit', 'swin', 'resnet']
    if args.arch_name == 'vit':
        batch_size_dict = {'small': 256, 'base': 256, 'large': 256}
    elif args.arch_name == 'swin':
        batch_size_dict = {'tiny': 128, 'small': 128, 'base': 64}
    elif args.arch_name == 'resnet':
        batch_size_dict = {'R50': 128, 'R101': 128, 'R152': 64}
    args.batch_size = batch_size_dict[args.net_type]
    if args.arch_name == 'vit':
        training_epochs_dict = {'small': 200, 'base': 100, 'large': 50}
    elif args.arch_name == 'swin':
        training_epochs_dict = {'tiny': 200, 'small': 100, 'base': 50}
    elif args.arch_name == 'resnet':
        training_epochs_dict = {'R50': 100, 'R101': 100, 'R152': 100}
    training_epochs = training_epochs_dict[args.net_type]
    if args.arch_name == 'vit':
        learning_rate_dict = {'small': 0.002, 'base': 0.0007, 'large': 0.0018}
    elif args.arch_name == 'swin':
        learning_rate_dict = {'tiny': 0.0014, 'small': 0.001, 'base': 0.0007}
    elif args.arch_name == 'resnet':
        learning_rate_dict = {'R50': 0.0014, 'R101': 0.001, 'R152': 0.0007}
    learning_rate = learning_rate_dict[args.net_type]
    if args.arch_name == 'vit':
        layer_decay_dict = {'small': 0.55, 'base': 0.4, 'large': 0.6}
    elif args.arch_name == 'swin':
        layer_decay_dict = {'tiny': 0.6, 'small': 0.4, 'base': 0.55}
    elif args.arch_name == 'resnet':
        layer_decay_dict = {'R50': 0.6, 'R101': 0.4, 'R152': 0.45}
    layer_decay = layer_decay_dict[args.net_type]
    warmup_epochs = int(training_epochs * 0.1)

    # network setting
    if args.arch_name == 'vit':
        drop_path_dict = {'small': 0.1, 'base': 0.2, 'large': 0.2}
    elif args.arch_name == 'swin':
        drop_path_dict = {'tiny': 0.1, 'small': 0.2, 'base': 0.2}
    elif args.arch_name == 'resnet':
        drop_path_dict = {'R50': None, 'R101': None, 'R152': None}
    args.drop_path = drop_path_dict[args.net_type]
    feature_model, _, cfg = setup_and_build_model(args)
    #feature_model.get_by_type(args.net_type)
    feature_model.use_mean_pooling = True
    embed_dim = feature_model.feat_dim
    actived_block_idx = feature_model.block_idx
    num_layers = feature_model.get_num_layers()

    train_transform = make_finetuning_transform(is_train=True, args=args)
    train_dataset = make_dataset(dataset_str=args.train_dataset_str, transform=train_transform)
    num_tasks = dist.get_world_size()
    global_rank = dist.get_rank()
    sampler_train = torch.utils.data.DistributedSampler(
        train_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
    logger.info("Sampler_train = %s" % str(sampler_train))
    val_transform = make_finetuning_transform(is_train=False, args=args)
    val_dataset = make_dataset(dataset_str=args.val_dataset_str,transform=val_transform)
    sampler_val = torch.utils.data.DistributedSampler(
        val_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=False
    )
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    val_data_loader = torch.utils.data.DataLoader(
        val_dataset, sampler=sampler_val,
        batch_size=int(1.5 * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    training_num_classes = len(torch.unique(torch.Tensor(train_dataset.get_targets().astype(int))))
    use_multi_stage_feat = args.arch_name == 'resnet'
    use_cls = args.arch_name  == 'vit'
    model = ModelWithClassifier(feature_model, embed_dim, training_num_classes, use_multi_stage_feat, use_cls)
    device = torch.device('cuda')
    model.to(device)
    model_ema = None  
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model, decay=args.model_ema_decay, device='', resume=''
        )
        print("Using EMA with decay = %.8f" % args.model_ema_decay)


    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_batch_size = args.batch_size * dist.get_world_size()
    num_training_steps_per_epoch = len(train_dataset) // total_batch_size
    model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
    model_without_ddp = model.module

    logger.info(f"Finetuning network: {args.net_type}")
    logger.info("Model = %s" % str(model_without_ddp))
    logger.info("Number of params: %d " % n_parameters)
    logger.info("LR = %.8f" % learning_rate)
    logger.info("Total batch size = %d" % total_batch_size)
    logger.info("Number of training examples = %d" % len(train_dataset))
    logger.info("Number of training step per epoch = %d" % num_training_steps_per_epoch)


    if layer_decay < 1.0 and args.arch_name == 'resnet':
        assigner = LayerDecayValueAssigner(
            list(layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)),
            prefix = 'feature_model', net_type='resnet', actived_block_idx=actived_block_idx, 
            depths=model_without_ddp.get_depths()
        )
    elif layer_decay < 1.0 and args.arch_name == 'swin':
        assigner = LayerDecayValueAssigner(
            list(layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)),
            prefix = 'feature_model', net_type='swin', actived_block_idx=actived_block_idx, 
            depths=model_without_ddp.get_depths()
        )
    elif layer_decay < 1.0 and args.arch_name == 'vit':
        assigner = LayerDecayValueAssigner(
            list(layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)), 
            prefix = 'feature_model', net_type='vit', actived_block_idx=actived_block_idx
        )
    else:
        assigner = None
    skip_weight_decay_list = model_without_ddp.no_weight_decay()
    optimizer = create_optimizer(
        args, model_without_ddp, skip_list=skip_weight_decay_list,
        get_num_layer=assigner.get_layer_id if assigner is not None else None,
        get_layer_scale=assigner.get_scale if assigner is not None else None)

    logger.info("Use step level LR scheduler!")
    lr_schedule_values = cosine_scheduler(
        learning_rate, args.min_lr, args.epochs, num_training_steps_per_epoch,
        warmup_epochs=warmup_epochs, warmup_steps=args.warmup_steps,
    )
    # mix up training
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        logger.info("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=training_num_classes)
    if mixup_fn is not None:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()
    logger.info("Criterion = %s" % str(criterion))

    logger.info(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    loss_scaler =  NativeScalerWithGradNormCount()
    for epoch in range(training_epochs):
        train_data_loader.sampler.set_epoch(epoch)
        # Training
        train_stats = train_one_epoch(
            model, criterion, train_data_loader, optimizer,
            device, epoch, loss_scaler, args.clip_grad, mixup_fn,
            start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values, wd_schedule_values=None,
            num_training_steps_per_epoch=num_training_steps_per_epoch, update_freq=1,
            model_ema=model_ema,
        )
        # Evaluation
        eval_model = model_ema.ema if args.model_ema else model
        test_stats = evaluate(val_data_loader, eval_model, device)
        logger.info(
            f"Accuracy of the network on the {len(val_dataset)} test images: {test_stats['acc1']:.1f}%"
        )
        if max_accuracy < test_stats["acc1"]:
            max_accuracy = test_stats["acc1"]
        logger.info(f'Epoch[{epoch + 1}/{training_epochs}] Max accuracy: {max_accuracy:.2f}%')
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}
        if args.output_dir and dist.get_rank() == 0:
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def main(args):
    run_finetnuing(args)
    return 0


if __name__ == "__main__":
    description = "Finetuning evaluation"
    args_parser = get_args_parser(description=description)
    args = args_parser.parse_args()
    sys.exit(main(args))


