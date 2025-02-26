# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging
import os
import random
import subprocess
from urllib.parse import urlparse
import re
import numpy as np
import torch
from torch import nn
from torch.distributed.checkpoint.state_dict import set_model_state_dict, StateDictOptions
logger = logging.getLogger("dinov2")

def revert_block_chunk_weight(state_dict):
    #convert blocks.chunkid.id.* to blocks.id.*: blocks.3.22. to blocks.22.
    return {re.sub(r'blocks\.(\d+)\.(\d+)\.', r'blocks.\2.', k): v for k, v in state_dict.items()}

def chunk_block_weight(state_dict, chunksize=6):
    #convert blocks.id.* to blocks.{id //(id//block_chunks)}.id.*: blocks.22. to blocks.3.22.
    return {re.sub(r'blocks\.(\d+)\.', lambda m: f'blocks.{int(m.group(1))//chunksize}.{m.group(1)}.', k): v for k, v in state_dict.items()}

def load_pretrained_weights(model, pretrained_weights, checkpoint_key, target_block_chunks=-1, wrapper_keys=["_orig_mod","backbone", "module"], device=None, strict=False):
    fsdp_compat_flag = False
    if urlparse(pretrained_weights).scheme:  # If it looks like an URL
        state_dict = torch.hub.load_state_dict_from_url(pretrained_weights, weights_only=True, map_location=device)
    else:
        try:
            state_dict = torch.load(pretrained_weights, weights_only=True, map_location=device)
        except Exception as e:
            state_dict = torch.load(pretrained_weights, weights_only=False, map_location=device)
            fsdp_compat_flag = True
    logger.info(f"Trying to load {pretrained_weights} with key {checkpoint_key}")
    if checkpoint_key is not None:
        if checkpoint_key is str and checkpoint_key in state_dict:
            logger.info(f"Take key {checkpoint_key} in provided checkpoint dict")
            state_dict = state_dict[checkpoint_key]
        else:
            for key in checkpoint_key:
                if key in state_dict:
                    logger.info(f"Take key {key} in provided checkpoint dict")
                    state_dict = state_dict[key]
                    break
    # remove `module.` prefix
    # remove `backbone.` prefix induced by multicrop wrapper
    for prefix in wrapper_keys:
        state_dict = {k.removeprefix(f"{prefix}."): v for k, v in state_dict.items()}
    if target_block_chunks>-1:
        state_dict = revert_block_chunk_weight(state_dict)
    if target_block_chunks > 0:
        logger.info(f"Chunking weights to  {target_block_chunks} blocks")
        state_dict = chunk_block_weight(state_dict, model.n_blocks//target_block_chunks)
    if fsdp_compat_flag:
        msg = set_model_state_dict(model, state_dict, options=StateDictOptions(strict=strict))
    else:
        msg = model.load_state_dict(state_dict, strict=strict)
    logger.info("Pretrained weights found at {} and loaded with msg: {}".format(pretrained_weights, msg))


def fix_random_seeds(seed=31):
    """
    Fix random seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_sha():
    cwd = os.path.dirname(os.path.abspath(__file__))

    def _run(command):
        return subprocess.check_output(command, cwd=cwd).decode("ascii").strip()

    sha = "N/A"
    diff = "clean"
    branch = "N/A"
    try:
        sha = _run(["git", "rev-parse", "HEAD"])
        subprocess.check_output(["git", "diff"], cwd=cwd)
        diff = _run(["git", "diff-index", "HEAD"])
        diff = "has uncommitted changes" if diff else "clean"
        branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    except Exception:
        pass
    message = f"sha: {sha}, status: {diff}, branch: {branch}"
    return message


class CosineScheduler(object):
    def __init__(self, base_value, final_value, total_iters, peak_iters=0, warmup_iters=0, start_warmup_value=0, freeze_iters=0, freeze_cut_iters=0):
        super().__init__()
        self.final_value = final_value
        self.total_iters = total_iters
        self.base_value = base_value
        self.freeze_iters = freeze_iters
        self.warmup_iters = warmup_iters
        self.peak_iters = peak_iters
        self.freeze_cut_iters = freeze_cut_iters
        self.start_warmup_value = start_warmup_value
    def __getitem__(self, it):
        if it < self.freeze_cut_iters: #cut to zero but don't effect other parts
            return 0
        elif it < self.freeze_iters: #freeze_schedule = np.zeros((freeze_iters))
            return 0
        elif  it < self.freeze_iters+self.warmup_iters:
            #warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)
            return self.start_warmup_value + (self.base_value - self.start_warmup_value) * (it - self.freeze_iters) / self.warmup_iters
        elif it < self.freeze_iters+self.warmup_iters+self.peak_iters:
            return self.base_value
        elif it < self.total_iters:
            # iters = np.arange(total_iters - warmup_iters - freeze_iters)
            # schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))
            decay_iters = self.freeze_iters + self.warmup_iters + self.peak_iters
            return self.final_value + 0.5 * (self.base_value - self.final_value) * (1 + np.cos(np.pi * (it - decay_iters) / (self.total_iters - decay_iters)))
        else:
            return self.final_value


class LinearScheduler(object):
    def __init__(self, base_value, final_value, total_iters, warmup_iters=0, start_warmup_value=0, freeze_iters=0):
        super().__init__()
        self.final_value = final_value
        self.total_iters = total_iters
        self.base_value = base_value
        self.warmup_iters = warmup_iters
        self.freeze_iters = freeze_iters
        self.start_warmup_value = start_warmup_value
    def __getitem__(self, it):
        if it < self.freeze_iters: #freeze_schedule = np.zeros((freeze_iters))
            return 0
        elif  it < self.freeze_iters+self.warmup_iters:
            #warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)
            return self.start_warmup_value + (self.base_value - self.start_warmup_value) * (it - self.freeze_iters) / self.warmup_iters
        elif it < self.total_iters:
            # iters = np.arange(total_iters - warmup_iters - freeze_iters)
            # schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))
            return self.base_value + (self.final_value - self.base_value) * (it - self.freeze_iters - self.warmup_iters) / (self.total_iters - self.freeze_iters - self.warmup_iters)
        else:
            return self.final_value

def has_batchnorms(model):
    bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)
    for name, module in model.named_modules():
        if isinstance(module, bn_types):
            return True
    return False
