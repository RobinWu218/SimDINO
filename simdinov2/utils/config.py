# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import math
import logging
import os

from omegaconf import OmegaConf

import simdinov2.distributed as dist
from simdinov2.logging import setup_logging
from simdinov2.utils import utils
from simdinov2.configs import load_config

logger = logging.getLogger("dinov2")


def apply_scaling_rules_to_cfg(cfg):  # to fix
    base_lr = cfg.optim.base_lr
    cfg.optim.lr = base_lr
    if cfg.optim.scaling_rule == "linear_wrt_1024":
        cfg.optim.lr *= cfg.train.batch_size_per_gpu * dist.get_global_size() / 1024.0
        logger.info(f"linear scaling learning rate; base: {base_lr}, new: {cfg.optim.lr}")
    elif cfg.optim.scaling_rule == "sqrt_wrt_1024":
        cfg.optim.lr *= math.sqrt(cfg.train.batch_size_per_gpu * dist.get_global_size() / 1024.0)
        logger.info(f"sqrt scaling learning rate; base: {base_lr}, new: {cfg.optim.lr}")
    else:
        raise NotImplementedError
    return cfg


def write_config(cfg, output_dir, name="config.yaml"):
    logger.info(OmegaConf.to_yaml(cfg))
    saved_cfg_path = os.path.join(output_dir, name)
    with open(saved_cfg_path, "w") as f:
        OmegaConf.save(config=cfg, f=f)
    return saved_cfg_path


def get_cfg_from_args(args):
    args.output_dir = os.path.abspath(args.output_dir)
    args.opts += [f"train.output_dir={args.output_dir}"]
    default_cfg = OmegaConf.create(load_config(args.base_config))
    cfg = OmegaConf.load(args.config_file)
    cfg = OmegaConf.merge(default_cfg, cfg, OmegaConf.from_cli(args.opts))
    return cfg


def default_setup(args, enable_dist: bool = True):
    if enable_dist:
        dist.enable(overwrite=True)
    seed = getattr(args, "seed", 0)
    rank = dist.get_global_rank()

    global logger
    setup_logging(output=args.output_dir, level=logging.INFO)
    logger = logging.getLogger("dinov2")

    utils.fix_random_seeds(seed + rank)
    logger.info("git:\n  {}\n".format(utils.get_sha()))
    logger.info("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))


class adict(dict):
    def __init__(self, iterable=None, **kwargs):#, _allow_non_exist=True
        if iterable is not None:
            for key, value in iterable.items():
                self.__setattr__(key, value)
        if kwargs:
            for key, value in kwargs.items():
                self.__setattr__(key, value)
        #self._allow_non_exist = _allow_non_exist
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            #if self._allow_non_exist:
            return None
            raise self.__attr_error(name)

    def __setattr__(self, name, value):
        if type(value) is dict:
            value = adict(value)
        self[name] = value

    def __delattr__(self, name):
        try:
            del self[name]
        except KeyError:
            raise self.__attr_error(name)

    def __attr_error(self, name):
        return AttributeError("type object '{subclass_name}' has no attribute '{attr_name}'".format(subclass_name=type(self).__name__, attr_name=name))

    def copy(self):
        return adict(self)
def setup(args, enable_dist=True):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg_from_args(args)
    os.makedirs(args.output_dir, exist_ok=True)
    default_setup(args, enable_dist)
    apply_scaling_rules_to_cfg(cfg)
    write_config(cfg, args.output_dir)
    cfg = adict(OmegaConf.to_object(cfg))
    return cfg
