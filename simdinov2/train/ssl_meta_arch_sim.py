# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

from functools import partial
import logging

import torch
from torch import nn

from simdinov2.loss import MCRLoss, DINOLoss, CosinePatchLoss, iBOTPatchLoss, KoLeoLoss
from simdinov2.models import build_model_from_cfg
from simdinov2.layers import DINOHead
from simdinov2.utils.utils import has_batchnorms
from simdinov2.utils.param_groups import get_params_groups_with_decay, fuse_params_groups
from simdinov2.fsdp import get_fsdp_wrapper, ShardedGradScaler, get_fsdp_modules, reshard_fsdp_model

from simdinov2.models.vision_transformer import BlockChunk
from simdinov2.utils.utils import load_pretrained_weights

logger = logging.getLogger("dinov2")

XFORMERS_AVAILABLE = False
try:
    from xformers.ops import fmha
    XFORMERS_AVAILABLE = True
except ImportError:
    pass

class SimSSLMetaArch(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.fp16_scaler = ShardedGradScaler() if cfg.compute_precision.grad_scaler else None

        student_model_dict = dict()
        teacher_model_dict = dict()

        student_backbone, teacher_backbone, embed_dim = build_model_from_cfg(cfg)
        student_model_dict["backbone"] = student_backbone
        teacher_model_dict["backbone"] = teacher_backbone
        logger.info(f"OPTIONS -- architecture : embed_dim: {embed_dim}")
        self.embed_dim = embed_dim

        self.do_dino = cfg.dino.loss_weight > 0
        self.do_koleo = cfg.dino.koleo_loss_weight > 0
        self.do_ibot = cfg.ibot.loss_weight > 0
        self.ibot_separate_head = cfg.ibot.separate_head
        
        self.dino_use_mcr = cfg.dino.use_mcr
        self.ibot_use_mcr = cfg.ibot.use_mcr
        self.drop_masks = cfg.student.drop_masks
        n_global_crops = 2
        #assert n_global_crops == 2
        n_local_crops = self.cfg.crops.local_crops_number
        #ncrops = n_global_crops + n_local_crops
        self.n_global_crops =n_global_crops
        self.n_local_crops = n_local_crops
        self.n_global_crops_loss_terms = (n_global_crops - 1) * n_global_crops
        self.n_total_crops_loss_terms = n_local_crops * n_global_crops + self.n_global_crops_loss_terms
        
        if self.do_dino:
            logger.info("OPTIONS -- DINO")
            logger.info(f"OPTIONS -- DINO -- loss_weight: {cfg.dino.loss_weight}")
            logger.info(f"OPTIONS -- DINO -- head_n_prototypes: {cfg.dino.head_n_prototypes}")
            logger.info(f"OPTIONS -- DINO -- head_bottleneck_dim: {cfg.dino.head_bottleneck_dim}")
            logger.info(f"OPTIONS -- DINO -- head_hidden_dim: {cfg.dino.head_hidden_dim}")
            self.dino_loss_weight = cfg.dino.loss_weight
            dino_head = partial(
                DINOHead,
                in_dim=embed_dim,
                out_dim=cfg.dino.head_n_prototypes,
                hidden_dim=cfg.dino.head_hidden_dim,
                bottleneck_dim=cfg.dino.head_bottleneck_dim,
                nlayers=cfg.dino.head_nlayers,
                normalize=cfg.dino.head_normalize,
                remove_last_layer=cfg.dino.remove_last_layer
            )
            dino_out_dim = cfg.dino.head_bottleneck_dim if cfg.dino.remove_last_layer else cfg.dino.head_n_prototypes
            self.dino_loss = MCRLoss(dino_out_dim, **cfg.dino.mcr) if self.dino_use_mcr else DINOLoss(dino_out_dim)
            if self.do_koleo:
                logger.info("OPTIONS -- DINO -- applying KOLEO regularization")
                self.koleo_loss = KoLeoLoss()
        else:
            logger.info("OPTIONS -- DINO -- not using DINO")
        
        if self.do_dino or (self.do_ibot and not self.ibot_separate_head):
            student_model_dict["dino_head"] = dino_head()
            teacher_model_dict["dino_head"] = dino_head()
        if self.do_ibot:
            logger.info("OPTIONS -- IBOT")
            logger.info(f"OPTIONS -- IBOT -- loss_weight: {cfg.ibot.loss_weight}")
            logger.info(f"OPTIONS -- IBOT masking -- ibot_mask_ratio_tuple: {cfg.ibot.mask_ratio_min_max}")
            logger.info(f"OPTIONS -- IBOT masking -- ibot_mask_sample_probability: {cfg.ibot.mask_sample_probability}")
            self.ibot_loss_weight = cfg.ibot.loss_weight
            assert max(cfg.ibot.mask_ratio_min_max) > 0, "please provide a positive mask ratio tuple for ibot"
            assert cfg.ibot.mask_sample_probability > 0, "please provide a positive mask probability for ibot"
            ibot_out_dim = (cfg.ibot.head_bottleneck_dim if cfg.dino.remove_last_layer else cfg.ibot.head_n_prototypes) if self.ibot_separate_head else dino_out_dim
            self.ibot_patch_loss = CosinePatchLoss(ibot_out_dim, **cfg.ibot.mcr) if self.ibot_use_mcr else iBOTPatchLoss(ibot_out_dim)
            if self.ibot_separate_head:
                if cfg.ibot.remove_last_layer:
                    logger.info("OPTIONS -- IBOT -- remove last layer")
                else:
                    logger.info(f"OPTIONS -- IBOT -- head_n_prototypes: {cfg.ibot.head_n_prototypes}")
                logger.info(f"OPTIONS -- IBOT -- head_bottleneck_dim: {cfg.ibot.head_bottleneck_dim}")
                logger.info(f"OPTIONS -- IBOT -- head_hidden_dim: {cfg.ibot.head_hidden_dim}")
                ibot_head = partial(
                    DINOHead,
                    in_dim=embed_dim,
                    out_dim=cfg.ibot.head_n_prototypes,
                    hidden_dim=cfg.ibot.head_hidden_dim,
                    bottleneck_dim=cfg.ibot.head_bottleneck_dim,
                    nlayers=cfg.ibot.head_nlayers,
                    normalize=cfg.ibot.head_normalize,
                    remove_last_layer=cfg.ibot.remove_last_layer
                )
                student_model_dict["ibot_head"] = ibot_head()
                teacher_model_dict["ibot_head"] = ibot_head()
            else:
                logger.info("OPTIONS -- IBOT -- head shared with DINO")

        self.need_to_synchronize_fsdp_streams = True

        self.student = nn.ModuleDict(student_model_dict)
        self.teacher = nn.ModuleDict(teacher_model_dict)

        if cfg.compile:
            self.teacher.compile()
            self.student.compile()
            getattr(self, "dino_loss", None) and self.dino_loss.compile()
            getattr(self, "koleo_loss", None) and self.koleo_loss.compile()
            getattr(self, "ibot_patch_loss", None) and self.ibot_patch_loss.compile()
        # there is no backpropagation through the teacher, so no need for gradients
        for p in self.teacher.parameters():
            p.requires_grad = False
        
        compiled_or_not = "" if cfg.compile else "not "
        logger.info(f"Student and Teacher are built: {cfg.student.arch} network {compiled_or_not}compiled.")

    def forward(self, inputs):
        raise NotImplementedError

    def backprop_loss(self, loss):
        if self.fp16_scaler is not None:
            self.fp16_scaler.scale(loss).backward()
        else:
            loss.backward()

    def forward_backward(self, images, teacher_temp, activate_ibot=True):
        n_global_crops = self.n_global_crops
        global_crops = images["collated_global_crops"].cuda(non_blocking=True)
        local_crops = images["collated_local_crops"].cuda(non_blocking=True)
        
        masks = images["collated_masks"].cuda(non_blocking=True)
        mask_indices_list = images["mask_indices_list"].cuda(non_blocking=True)
        n_masked_patches_tensor = images["n_masked_patches"].cuda(non_blocking=True)
        n_masked_patches = mask_indices_list.shape[0]
        #upperbound = images["upperbound"] #upperbound逻辑和修改后逻辑一致, 可以在多机训练中带来内存对齐但效率提升待确认
        masks_weight = images["masks_weight"].cuda(non_blocking=True)
        
        do_ibot = self.do_ibot
        
        # teacher output
        @torch.no_grad()
        def get_teacher_output():
            teacher_backbone_output_dict = self.teacher.backbone(global_crops, is_training=True)
            teacher_cls_tokens = teacher_backbone_output_dict["x_norm_clstoken"]
            teacher_patch_tokens = teacher_backbone_output_dict["x_norm_patchtokens"]
            _dim = teacher_patch_tokens.shape[-1]
            n_cls_tokens = teacher_cls_tokens.shape[0]

            if do_ibot and not self.ibot_separate_head:
                buffer_tensor_teacher = teacher_patch_tokens.new_zeros(n_masked_patches + n_cls_tokens, _dim)
                buffer_tensor_teacher[:n_cls_tokens].copy_(teacher_cls_tokens)
                torch.index_select(
                    teacher_patch_tokens.flatten(0, 1),
                    dim=0,
                    index=mask_indices_list,
                    out=buffer_tensor_teacher[n_cls_tokens : n_cls_tokens + n_masked_patches],
                )
                tokens_after_head = self.teacher.dino_head(buffer_tensor_teacher)
                teacher_cls_tokens_after_head, masked_teacher_patch_tokens_after_head = tokens_after_head.split([n_cls_tokens, n_masked_patches])
            elif do_ibot and self.ibot_separate_head:
                teacher_cls_tokens_after_head = self.teacher.dino_head(teacher_cls_tokens)
                buffer_tensor_teacher = teacher_patch_tokens.new_zeros(n_masked_patches, _dim)
                torch.index_select(
                    teacher_patch_tokens.flatten(0, 1),
                    dim=0,
                    index=mask_indices_list,
                    out=buffer_tensor_teacher,
                )
                masked_teacher_patch_tokens_after_head = self.teacher.ibot_head(buffer_tensor_teacher)
            else:
                teacher_cls_tokens_after_head = self.teacher.dino_head(teacher_cls_tokens)
                masked_teacher_patch_tokens_after_head = None

            masked_teacher_ibot_softmaxed_centered = None
            if self.cfg.train.centering == "centering":
                teacher_dino_softmaxed_centered_list = self.dino_loss.softmax_center_teacher(
                    teacher_cls_tokens_after_head, teacher_temp=teacher_temp
                ).view(n_global_crops, -1, *teacher_cls_tokens_after_head.shape[1:])
                self.dino_loss.update_center(teacher_cls_tokens_after_head)
                if do_ibot:
                    masked_teacher_patch_tokens_after_head = masked_teacher_patch_tokens_after_head.unsqueeze(0)
                    masked_teacher_ibot_softmaxed_centered = self.ibot_patch_loss.softmax_center_teacher(
                        masked_teacher_patch_tokens_after_head[:, :n_masked_patches], teacher_temp=teacher_temp
                    )
                    masked_teacher_ibot_softmaxed_centered = masked_teacher_ibot_softmaxed_centered.squeeze(0)
                    self.ibot_patch_loss.update_center(masked_teacher_patch_tokens_after_head[:n_masked_patches])
            elif self.cfg.train.centering == "sinkhorn_knopp":
                teacher_dino_softmaxed_centered_list = self.dino_loss.sinkhorn_knopp_teacher(
                    teacher_cls_tokens_after_head, teacher_temp=teacher_temp
                ).view(n_global_crops, -1, *teacher_cls_tokens_after_head.shape[1:])

                if do_ibot:
                    masked_teacher_ibot_softmaxed_centered = self.ibot_patch_loss.sinkhorn_knopp_teacher(
                        masked_teacher_patch_tokens_after_head,
                        teacher_temp=teacher_temp,
                        n_masked_patches_tensor=n_masked_patches_tensor,
                    )
            else:
                teacher_dino_softmaxed_centered_list = teacher_cls_tokens_after_head
                masked_teacher_ibot_softmaxed_centered = masked_teacher_patch_tokens_after_head
            return teacher_cls_tokens, teacher_dino_softmaxed_centered_list, masked_teacher_ibot_softmaxed_centered

        teacher_cls_tokens, teacher_dino_softmaxed_centered_list, masked_teacher_ibot_softmaxed_centered = get_teacher_output()
        reshard_fsdp_model(self.teacher)

        loss_dict = {}

        loss_accumulator = 0  # for backprop
        if "nested" in self.cfg.student.block: #nested computation tricks
            student_global_backbone_output_dict, student_local_backbone_output_dict = self.student.backbone(
                [global_crops, local_crops], masks=[masks, None], is_training=True
            )
        else:
            student_global_backbone_output_dict = self.student.backbone(global_crops, masks=masks, is_training=True)
            student_local_backbone_output_dict = self.student.backbone(local_crops, is_training=True)

        inputs_for_student_head = []

        # 1a: local crops cls tokens
        student_local_cls_tokens = student_local_backbone_output_dict["x_norm_clstoken"]
        inputs_for_student_head.append(student_local_cls_tokens)

        # 1b: global crops cls tokens
        student_global_cls_tokens = student_global_backbone_output_dict["x_norm_clstoken"]
        inputs_for_student_head.append(student_global_cls_tokens)

        # 1c: global crops patch tokens
        if do_ibot:
            ibot_student_patch_tokens = student_global_backbone_output_dict["x_norm_patchtokens"]
            if self.drop_masks:
                raise NotImplementedError("Drop masks not implemented for ibot, need decoder like MAE")
            else:
                buffer_tensor_patch_tokens=torch.index_select(ibot_student_patch_tokens.flatten(0, 1), dim=0, index=mask_indices_list)
            if not self.ibot_separate_head:
                inputs_for_student_head.append(buffer_tensor_patch_tokens)
            else:
                student_global_masked_patch_tokens_after_head = self.student.ibot_head(buffer_tensor_patch_tokens)
        del student_global_backbone_output_dict, student_local_backbone_output_dict

        if self.do_dino and self.do_koleo:
            koleo_loss = self.cfg.dino.koleo_loss_weight * sum(
                self.koleo_loss(p) for p in student_global_cls_tokens.chunk(2)
            )  # we don't apply koleo loss between cls tokens of a same image
            loss_accumulator += koleo_loss
            loss_dict["koleo_loss"] = (
                koleo_loss #/ n_global_crops
            )  # this is to display the same losses as before but we can remove eventually
            
        #del student_global_cls_tokens, student_local_cls_tokens
        # 2: run
        if XFORMERS_AVAILABLE:
            _attn_bias, cat_inputs = fmha.BlockDiagonalMask.from_tensor_list([x.unsqueeze(0) for x in inputs_for_student_head])
            outputs_list = [x.squeeze(0) for x in _attn_bias.split(self.student.dino_head(cat_inputs))]
            del _attn_bias, cat_inputs
        else:
            seqs = [x.shape[0] for x in inputs_for_student_head]
            inputs_for_student_head = torch.cat(inputs_for_student_head)
            outputs_list = self.student.dino_head(inputs_for_student_head).split(seqs)
        del inputs_for_student_head
        if do_ibot and not self.ibot_separate_head:
            student_local_cls_tokens_after_head, student_global_cls_tokens_after_head, student_global_masked_patch_tokens_after_head = outputs_list
        else:
            student_local_cls_tokens_after_head,student_global_cls_tokens_after_head = outputs_list
        del outputs_list
        if self.do_dino:
            # compute loss
            dino_crops_loss, dino_loss_dict = self.dino_loss(
                student_global_cls_tokens_after_head.chunk(2) + student_local_cls_tokens_after_head.chunk(self.n_local_crops),
                teacher_dino_softmaxed_centered_list.chunk(2), no_diag=True
            )
            if self.dino_use_mcr:
                dino_loss_dict = {"dino_mcr_"+k: v for k, v in dino_loss_dict.items()}
                loss_dict |= dino_loss_dict
            else:
                #dino loss averaged over the number of crops
                dino_crops_loss /= self.n_total_crops_loss_terms
                loss_dict["dino_loss"] = dino_crops_loss
                loss_dict["dino_global_crops_loss"] = dino_loss_dict / self.n_global_crops_loss_terms
            # accumulate loss
            loss_accumulator += self.dino_loss_weight * dino_crops_loss
        del student_global_cls_tokens_after_head, student_local_cls_tokens_after_head
        del teacher_dino_softmaxed_centered_list
        if do_ibot:
            # compute loss
            ibot_patch_loss = self.ibot_patch_loss.forward_masked(
                    student_global_masked_patch_tokens_after_head,
                    masked_teacher_ibot_softmaxed_centered,
                    student_masks_flat=masks,
                    n_masked_patches=n_masked_patches,
                    masks_weight=masks_weight,
                )

            if self.ibot_use_mcr:
                ibot_patch_loss, ibot_loss_dict = ibot_patch_loss
                ibot_loss_dict = {"ibot_"+k: v for k, v in ibot_loss_dict.items()}
                loss_dict |= ibot_loss_dict
            else:
                loss_dict["ibot_loss"] = ibot_patch_loss # / n_global_crops

            # accumulate loss
            loss_accumulator += self.ibot_loss_weight * ibot_patch_loss

        loss_dict["total_loss"] = loss_accumulator.detach()
        self.backprop_loss(loss_accumulator)

        self.fsdp_synchronize_streams()

        if torch.isnan(loss_accumulator):
            print(f"loss_accumulator NaN detected: {loss_dict}")
            import debugpy
            debugpy.breakpoint()
        return loss_dict

    def fsdp_synchronize_streams(self):
        if self.need_to_synchronize_fsdp_streams:
            torch.cuda.synchronize()
            # self.student.dino_head._streams = (
            #     self.teacher.dino_head._streams
            # ) = self.student.backbone._streams = self.teacher.backbone._streams
            
            for attr in {"_unshard_stream", "_post_backward_stream", "_pre_unshard_stream", "_all_reduce_stream", "_default_stream"}:
                stream = getattr(self.teacher.backbone, attr)
                setattr(self.student.dino_head, attr, stream)
                setattr(self.teacher.dino_head, attr, stream)
                setattr(self.student.backbone, attr, stream)
            self.need_to_synchronize_fsdp_streams = False

    def update_teacher(self, m):
        if m == 1.0:
            return
        elif m == 0.0:
            self.teacher.load_state_dict(self.student.state_dict())
            return
        student_param_list = []
        teacher_param_list = []
        with torch.no_grad():
            for k in self.student.keys():
                for ms, mt in zip(get_fsdp_modules(self.student[k]), get_fsdp_modules(self.teacher[k])):
                    student_param_list += ms.params
                    teacher_param_list += mt.params
            if hasattr(torch, '_foreach_lerp_'):
                torch._foreach_lerp_(teacher_param_list, student_param_list, weight=1. - m)
            else:
                torch._foreach_mul_(teacher_param_list, m)
                torch._foreach_add_(teacher_param_list, student_param_list, alpha=1. - m)

    def train(self):
        super().train()
        self.teacher.eval()

    def get_maybe_fused_params_for_submodel(self, m):
        params_groups = get_params_groups_with_decay(
            model=m,
            lr_decay_rate=self.cfg.optim.layerwise_decay,
            patch_embed_lr_mult=self.cfg.optim.patch_embed_lr_mult,
        )
        fused_params_groups = fuse_params_groups(params_groups)
        logger.info("fusing param groups")

        for g in fused_params_groups:
            g["foreach"] = True
        return fused_params_groups

    def get_params_groups(self, fused=True):
        all_params_groups = []
        for m in self.student.values():
            all_params_groups += self.get_maybe_fused_params_for_submodel(m) if fused else get_params_groups_with_decay(
                model=m,
                lr_decay_rate=self.cfg.optim.layerwise_decay,
                patch_embed_lr_mult=self.cfg.optim.patch_embed_lr_mult,
            )
        return all_params_groups

    def prepare_for_distributed_training(self):
        logger.info("DISTRIBUTED FSDP -- preparing model for distributed training")
        if has_batchnorms(self.student):
            raise NotImplementedError
        # below will synchronize all student subnetworks across gpus:
        for k, v in self.student.items():
            self.teacher[k].load_state_dict(self.student[k].state_dict())
            student_model_cfg = self.cfg.compute_precision.student[k]
            self.student[k] = get_fsdp_wrapper(student_model_cfg, modules_to_wrap={BlockChunk})(self.student[k])
            teacher_model_cfg = self.cfg.compute_precision.teacher[k]
            self.teacher[k] = get_fsdp_wrapper(teacher_model_cfg, modules_to_wrap={BlockChunk})(self.teacher[k])
