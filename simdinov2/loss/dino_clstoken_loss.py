# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

class DINOCenter(nn.Module):
    def __init__(
        self,
        out_dim,
        enable=True,
        center_momentum=0.9,
    ):
        super().__init__()
        if not enable:
            return
        self.register_buffer("center", torch.zeros(1, out_dim))
        self.center_momentum = center_momentum
        self.updated = True
        self.reduce_handle = None
        self.len_teacher_output = None
        self.async_batch_center = None

    @torch.no_grad()
    def softmax_center_teacher(self, teacher_output, teacher_temp):
        self.apply_center_update()
        # teacher centering and sharpening
        return F.softmax((teacher_output - self.center) / teacher_temp, dim=-1)

    @torch.no_grad()
    def sinkhorn_knopp_teacher(self, teacher_output, teacher_temp, n_iterations=3):
        teacher_output = teacher_output.float()
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        Q = torch.exp(teacher_output / teacher_temp).t()  # Q is K-by-B for consistency with notations from our paper
        B = Q.shape[1] * world_size  # number of samples to assign
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
    def update_center(self, teacher_output):
        self.reduce_center_update(teacher_output)

    @torch.no_grad()
    def reduce_center_update(self, teacher_output):
        self.updated = False
        self.len_teacher_output = len(teacher_output)
        self.async_batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        if dist.is_initialized():
            self.reduce_handle = dist.all_reduce(self.async_batch_center, async_op=True)

    @torch.no_grad()
    def apply_center_update(self):
        if self.updated is False:
            world_size = dist.get_world_size() if dist.is_initialized() else 1

            if self.reduce_handle is not None:
                self.reduce_handle.wait()
            _t = self.async_batch_center / (self.len_teacher_output * world_size)

            self.center = self.center * self.center_momentum + _t * (1 - self.center_momentum)

            self.updated = True

def half_logdet(X):
    return torch.linalg.cholesky_ex(X)[0].diagonal().log().sum()

class MCRLoss(DINOCenter):
    def __init__(self, out_dim, expa_type=0, reduce_cov=0, eps=0.05, coeff=1, center=False, *args, **kwargs):
        super().__init__(out_dim, enable=center)
        self.eps = eps
        self.coeff = coeff
        self.expa_type = expa_type
        self.reduce_cov = reduce_cov
    
    def forward(self, student_feat_list, teacher_feat_list, no_diag=True, normalized=True):
        """
        Expansion Loss and Compression Loss between features of the teacher and student networks.
        """
        # Convert lists of tensors to a single tensor for vectorized operations
        student_feat = torch.stack(student_feat_list) #ncrops,N,D
        teacher_feat = torch.stack(teacher_feat_list) #2,N,D
        if not normalized:
            student_feat = F.normalize(student_feat, p=2, dim=-1)
            teacher_feat = F.normalize(teacher_feat, p=2, dim=-1)
        comp_loss, global_comp_loss = self.calc_compression(student_feat, teacher_feat, no_diag=no_diag)
        match self.expa_type:
            case 0:  # only compute expansion on global views
                expa_feat = student_feat[:len(teacher_feat)]
            case 1:  # center with teacher
                expa_feat = (student_feat[:len(teacher_feat)] + teacher_feat) / 2
        expa_loss = self.calc_expansion(expa_feat)
        loss = - self.coeff * comp_loss - expa_loss
        return loss, {"loss": loss.detach(), "comp_loss":comp_loss.detach(), "global_comp_loss":global_comp_loss.detach(), "expa_loss":expa_loss.detach()}
    
    def calc_compression(self, student_feat_list, teacher_feat_list, no_diag=True):
        """
        Compute compression loss between student and teacher features.
        """

        # Compute cosine similarity for all pairs
        comp_loss = 0
        sim = (teacher_feat_list.unsqueeze(1)*student_feat_list.unsqueeze(0)).sum(-1).mean(-1)
        # Mask out the diagonal elements where student and teacher operate on the same view
        #mask = torch.eye(len(teacher_feat_list), len(student_feat_list), dtype=torch.bool,device=cosine_sim.device).unsqueeze_(2)
        #sim = cosine_sim.masked_fill(mask, 0)
        if no_diag:
            sim.view(-1)[:: (len(student_feat_list) + 1)].fill_(0)  # Trick to fill diagonal
        
        n_loss_terms = len(teacher_feat_list)* len(student_feat_list) - min(len(teacher_feat_list), len(student_feat_list))
        # Sum the cosine similarities
        comp_loss = sim.sum()/n_loss_terms
        global_comp_loss = sim[:, :len(teacher_feat_list)].detach().sum().div_(len(teacher_feat_list))
        return comp_loss, global_comp_loss

    def calc_expansion(self, feat_list, cross_list=None) -> torch.Tensor:
        """
        Compute expansion loss using Coding Rate estimation.
        """
        cov = []
        num_views = len(feat_list)
        m, p = feat_list[0].shape
        cov = torch.einsum('nbc,nbd->ncd', feat_list, cross_list or feat_list)
        N=1
        if dist.is_initialized():
            N = dist.get_world_size()
            if self.reduce_cov == 1:
                cov = dist.nn.all_reduce(cov)
        loss = 0
        scalar =  p / (m * N * self.eps)
        I = torch.eye(p, device=cov[0].device)
        loss = sum([half_logdet(I + scalar * cov[i]) for i in range(num_views)])
        #loss =  torch.logdet(I + scalar * cov).sum()/2
        loss /= num_views
        loss *= (p+N*m)/(p*N*m) # the balancing factor gamma, you can also use the next line. This is ultimately a heuristic, so feel free to experiment.
        # loss *= ((self.eps * N * m) ** 0.5 / p)
        return loss

class DINOLoss(DINOCenter):
    def __init__(
        self,
        out_dim,
        student_temp=0.1,
    ):
        super().__init__(out_dim)
        self.student_temp = student_temp
        self.reduce_handle = None
        self.len_teacher_output = None
        self.async_batch_center = None
        
    def forward(self, student_output_list, teacher_out_softmaxed_centered_list, no_diag=False):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        # TODO: Use cross_entropy_distribution here
        total_loss = 0
        global_loss = 0
        for i, s in enumerate(student_output_list):
            lsm = F.log_softmax(s / self.student_temp, dim=-1)
            for j, t in enumerate(teacher_out_softmaxed_centered_list):
                if no_diag and i == j:
                        continue
                loss = torch.sum(t * lsm, dim=-1).mean()
                total_loss -= loss
                if no_diag and i < len(teacher_out_softmaxed_centered_list):
                    global_loss -= loss.detach()
        return total_loss, global_loss
