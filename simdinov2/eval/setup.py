# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import argparse
from typing import Any, List, Optional, Tuple

import torch
import torch.backends.cudnn as cudnn

from simdinov2.models import build_model_from_cfg
from simdinov2.utils.config import setup
from simdinov2.utils.utils import load_pretrained_weights
import logging
logger = logging.getLogger("dinov2")


def get_args_parser(
    description: Optional[str] = None,
    parents: Optional[List[argparse.ArgumentParser]] = None,
    add_help: bool = True,
):
    parser = argparse.ArgumentParser(
        description=description,
        parents=parents or [],
        add_help=add_help,
    )
    parser.add_argument("--base-config", default="ssl_default_config", metavar="FILE", help="path to base config file")
    parser.add_argument(
        "--config-file",
        type=str,
        help="Model configuration file",
    )
    parser.add_argument(
        "--pretrained-weights",
        type=str,
        help="Pretrained model weights",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        type=str,
        help="Output directory to write results and logs",
    )
    parser.add_argument(
        "--opts",
        help="Extra configuration options",
        default=[],
        nargs="+",
    )
    return parser


def get_autocast_dtype(config):
    teacher_dtype_str = config.compute_precision.teacher.backbone.mixed_precision.param_dtype
    if teacher_dtype_str == "fp16":
        return torch.half
    elif teacher_dtype_str == "bf16":
        return torch.bfloat16
    else:
        return torch.float


def build_model(cfg, pretrained_weights, is_train=False, target_block_chunks=-2):
    cfg.student.pretrained_weights = ""
    model, _ = build_model_from_cfg(cfg, only_teacher=True)
    if target_block_chunks<-1:
        target_block_chunks = cfg.student.block_chunks
    load_pretrained_weights(model, pretrained_weights, ("model", "teacher"), target_block_chunks )
    if is_train:
        model.train()
    else:
        model.eval()
    model.cuda()
    return model


def setup_and_build_model(args, is_train=False) -> Tuple[Any, torch.dtype]:
    cudnn.benchmark = True
    cfg = setup(args)
    model = build_model(cfg, args.pretrained_weights, is_train)
    if cfg.evaluation.patch_size is not None and cfg.evaluation.patch_size!=cfg.student.patch_size:
        logger.info(f"OPTIONS -- evaluation patch size: resizing from {cfg.student.patch_size} to {cfg.evaluation.patch_size}")
        model.update_patch_size(cfg.evaluation.patch_size)
    if cfg.evaluation.img_size is not None and cfg.evaluation.img_size!=cfg.crops.global_crops_size:
        logger.info(f"OPTIONS -- evaluation img size: resizing from {cfg.crops.global_crops_size} to {cfg.evaluation.img_size}")
        model.update_img_size(cfg.evaluation.img_size)
    autocast_dtype = get_autocast_dtype(cfg)
    return model, autocast_dtype, cfg
