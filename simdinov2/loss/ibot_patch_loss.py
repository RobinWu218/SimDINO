# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
import numpy as np
import logging


logger = logging.getLogger("dinov2")


try:
    from xformers.ops.common import cross_entropy
    #from flash_attn.ops.triton.cross_entropy import cross_entropy_loss
    def lossfunc(s:torch.Tensor, t:torch.Tensor, temp):
        s = s.float()
        t = t.float()
        if s.ndim == 2:
            return -cross_entropy(s.unsqueeze(0), t.unsqueeze(0), temp, bw_inplace=True).squeeze(0)
        elif s.ndim == 3:
            return -cross_entropy(s, t, temp, bw_inplace=True)
except ImportError:
    try:
        from .fused_ce_loss import cross_entropy
        def lossfunc(s, t, temp):
            return -cross_entropy(s/temp, t, reduction="none", chunksize=256)
    except ImportError:
        def lossfunc(s, t, temp):
            return F.cross_entropy(s/temp, t, reduction="none")
            #return torch.sum(t * F.log_softmax(s / temp, dim=-1), dim=-1)


class PatchDINOCenter(nn.Module):
    def __init__(
        self,
        patch_out_dim,
        enable=True,
        center_momentum=0.9,
    ):
        super().__init__()
        if not enable:
            return
        self.register_buffer("center", torch.zeros(1, 1, patch_out_dim))
        self.center_momentum = center_momentum
        self.updated = True
        self.reduce_handle = None
        self.len_teacher_patch_tokens = None
        self.async_batch_center = None

    @torch.no_grad()
    def softmax_center_teacher(self, teacher_patch_tokens, teacher_temp):
        self.apply_center_update()
        # teacher centering and sharpening
        #
        # WARNING:
        #   as self.center is a float32, everything gets casted to float32 afterwards
        #
        # teacher_patch_tokens = teacher_patch_tokens.float()
        # return F.softmax((teacher_patch_tokens.sub_(self.center.to(teacher_patch_tokens.dtype))).mul_(1 / teacher_temp), dim=-1)

        return F.softmax((teacher_patch_tokens - self.center) / teacher_temp, dim=-1)

        # this is experimental, keep everything in float16 and let's see what happens:
        # return F.softmax((teacher_patch_tokens.sub_(self.center)) / teacher_temp, dim=-1)

    @torch.no_grad()
    def sinkhorn_knopp_teacher(self, teacher_output, teacher_temp, n_masked_patches_tensor, n_iterations=3):
        teacher_output = teacher_output.float()
        # world_size = dist.get_world_size() if dist.is_initialized() else 1
        Q = torch.exp(teacher_output / teacher_temp).t()  # Q is K-by-B for consistency with notations from our paper
        # B = Q.shape[1] * world_size # number of samples to assign
        B = n_masked_patches_tensor
        dist.all_reduce(B)
        K = Q.shape[0]  # how many prototypes

        # make the matrix sums to 1
        sum_Q = torch.sum(Q)
        if dist.is_initialized():
            dist.all_reduce(sum_Q)
        Q /= sum_Q

        for it in range(n_iterations):
            # normalize each row: total weight per prototype must be 1/K
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            if dist.is_initialized():
                dist.all_reduce(sum_of_rows)
            Q /= sum_of_rows
            Q /= K

            # normalize each column: total weight per sample must be 1/B
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B

        Q *= B  # the columns must sum to 1 so that Q is an assignment
        return Q.t()

    @torch.no_grad()
    def update_center(self, teacher_patch_tokens):
        self.reduce_center_update(teacher_patch_tokens)

    @torch.no_grad()
    def reduce_center_update(self, teacher_patch_tokens):
        self.updated = False
        self.len_teacher_patch_tokens = len(teacher_patch_tokens)
        self.async_batch_center = torch.sum(teacher_patch_tokens.mean(1), dim=0, keepdim=True)
        if dist.is_initialized():
            self.reduce_handle = dist.all_reduce(self.async_batch_center, async_op=True)

    @torch.no_grad()
    def apply_center_update(self):
        if self.updated is False:
            world_size = dist.get_world_size() if dist.is_initialized() else 1

            if self.reduce_handle is not None:
                self.reduce_handle.wait()
            _t = self.async_batch_center / (self.len_teacher_patch_tokens * world_size)

            self.center = self.center * self.center_momentum + _t * (1 - self.center_momentum)

            self.updated = True

class CosinePatchLoss(PatchDINOCenter):
    def __init__(self, patch_out_dim, center=False, center_momentum=0.9, **kwargs):
        super().__init__(patch_out_dim, center, center_momentum)
    
    def forward(self, student_patch_tokens, teacher_patch_tokens, student_masks_flat):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        student_patch_tokens: (B, N, D) tensor
        teacher_patch_tokens: (B, N, D) tensor
        student_masks_flat: (B, N) tensor
        """
        loss = F.cosine_similarity(teacher_patch_tokens, student_patch_tokens, dim=-1)
        loss = torch.sum(loss * student_masks_flat.float(), dim=-1) / student_masks_flat.sum(dim=-1).clamp(min=1.0)
        comp_loss= -loss.mean()
        return comp_loss, {"comp_loss": comp_loss.detach()}

    def forward_masked(
        self,
        student_patch_tokens_masked,
        teacher_patch_tokens_masked,
        student_masks_flat,
        n_masked_patches=None,
        masks_weight=None,
    ):
        loss = F.cosine_similarity(teacher_patch_tokens_masked, student_patch_tokens_masked, dim=-1)
        if masks_weight is None:
            masks_weight = (
                (1 / student_masks_flat.sum(-1).clamp(min=1.0))
                .unsqueeze(-1)
                .expand_as(student_masks_flat)[student_masks_flat]
            )
        if n_masked_patches is not None:
            loss = loss[:n_masked_patches]
        loss = loss * masks_weight
        comp_loss = -loss.sum() / student_masks_flat.shape[0]
        return comp_loss, {"comp_loss": comp_loss.detach()}

class iBOTPatchLoss(PatchDINOCenter):
    def __init__(self, patch_out_dim, student_temp=0.1, center_momentum=0.9, **kwargs):
        super().__init__(patch_out_dim, center_momentum)
        self.student_temp = student_temp

    def forward(self, student_patch_tokens, teacher_patch_tokens, student_masks_flat):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        student_patch_tokens: (B, N, D) tensor
        teacher_patch_tokens: (B, N, D) tensor
        student_masks_flat: (B, N) tensor
        """
        t = teacher_patch_tokens
        s = student_patch_tokens
        loss = torch.sum(t * F.log_softmax(s / self.student_temp, dim=-1), dim=-1)
        loss = torch.sum(loss * student_masks_flat.float(), dim=-1) / student_masks_flat.sum(dim=-1).clamp(min=1.0)
        return -loss.mean()

    def forward_masked(
        self,
        student_patch_tokens_masked,
        teacher_patch_tokens_masked,
        student_masks_flat,
        n_masked_patches=None,
        masks_weight=None,
    ):
        t = teacher_patch_tokens_masked
        s = student_patch_tokens_masked
        # loss = torch.sum(t * F.log_softmax(s / self.student_temp, dim=-1), dim=-1)
        loss = lossfunc(s, t, self.student_temp)
        if masks_weight is None:
            masks_weight = (
                (1 / student_masks_flat.sum(-1).clamp(min=1.0))
                .unsqueeze(-1)
                .expand_as(student_masks_flat)[student_masks_flat]
            )
        if n_masked_patches is not None:
            loss = loss[:n_masked_patches]
        loss = loss * masks_weight
        return -loss.sum() / student_masks_flat.shape[0]
