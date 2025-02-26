"""This is a fused cross-entropy and linear layer. Idea is copied 
from https://github.com/linkedin/Liger-Kernel who just copied it from
https://github.com/mgmalek/efficient_cross_entropy
"""

import torch
from torch.autograd import Function
from torch.nn import functional as F


class EfficientCrossEntropy(Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor, target: torch.Tensor, chunksize = 2048,
                reduction: str = "mean", label_smoothing: float = 0.0, inplace_backward: bool = True):
        if label_smoothing > 0.0:
            raise NotImplementedError("Label smoothing is not implemented yet.")
        bs = input.shape[0]
        needs_grad = ctx.needs_input_grad[0]
        if needs_grad:
            act_grad = torch.empty_like(input)
        if reduction == "none":
            out_loss = torch.empty(bs, device=input.device)
        else:
            out_loss = torch.tensor(0.0, device=input.device)
        is_label = len(target.shape) == 1
        for b in range(0, bs, chunksize):
            end_idx = min(b + chunksize, bs)

            # Get current batch chunks
            logits_chunk = input[b:end_idx]  # [chunk_size, V]
            target_chunk = target[b:end_idx]  # [chunk_size] if is_label else [chunk_size, V]

            # Compute softmax and loss
            max_logits = torch.max(logits_chunk, dim=-1, keepdim=True)[0]
            logits_chunk -= max_logits
            exp_logits = torch.exp(logits_chunk)
            sum_exp = torch.sum(exp_logits, dim=-1, keepdim=True)
            LSE_minus_max = torch.log(sum_exp)
            
            if is_label:
                # Compute loss using gather
                correct_logits = torch.gather(
                    logits_chunk, 1, target_chunk.unsqueeze(1)
                )  # [chunk_size, 1]
                if reduction == "none":
                    out_loss[b:end_idx] = LSE_minus_max.squeeze() + correct_logits.squeeze()
                else:
                    out_loss += torch.sum(LSE_minus_max.squeeze() - correct_logits.squeeze())
            else: #target is probs
                #out_loss -= torch.sum(target_chunk * torch.log(probs))
                #torch.log(probs)=logits_chunk-LSE
                if reduction == "none":
                    out_loss[b:end_idx] = torch.sum(target_chunk * (LSE_minus_max-logits_chunk), dim=-1)
                else:
                    out_loss += torch.sum(target_chunk * (LSE_minus_max-logits_chunk))
                
            # Compute gradients
            if needs_grad:
                probs = exp_logits / sum_exp  # [chunk_size, V]
                if is_label:
                    grad = probs.clone()  # [chunk_size, V]
                    grad.scatter_(
                        1,
                        target_chunk.unsqueeze(1),
                        grad.gather(1, target_chunk.unsqueeze(1)) - 1,
                    )
                else: #target is probs, grad to input is:
                    grad = - target_chunk + torch.sum(target_chunk, dim=-1, keepdim=True) * probs

                # Accumulate gradients
                act_grad[b:end_idx] = grad  # [chunk_size, V]

        # Scale
        if reduction == "mean":
            scale = 1.0 / bs
        else:
            scale = 1.0
        if needs_grad:
            act_grad *= scale
            ctx.save_for_backward(act_grad)
            ctx.inplace_backward = inplace_backward
        return scale * out_loss

    @staticmethod
    def backward(ctx, grad_output:torch.Tensor):  # type: ignore
        (act_grad,) = ctx.saved_tensors
        #make sure grad_output have same dim as act_grad, or unsqueeze it
        if grad_output.dim() == 1:
            return act_grad.mul_(grad_output.unsqueeze(-1)), None, None, None, None, None
        return act_grad.mul_(grad_output), None, None, None, None, None


class EfficientCrossEntropyFused(Function):
    @staticmethod
    def forward(ctx, weight: torch.Tensor, act: torch.Tensor, labels: torch.Tensor):
        bs = act.shape[0]
        weight_grad = torch.zeros_like(weight)
        act_grad = torch.empty_like(act)
        out_loss = torch.tensor(0.0, device=act.device)
        chunksize = 2048

        for b in range(0, bs, chunksize):
            end_idx = min(b + chunksize, bs)

            # Get current batch chunks
            act_chunk = act[b:end_idx]  # [chunk_size, H]
            labels_chunk = labels[b:end_idx]  # [chunk_size]

            # Compute logits
            logits = F.linear(act_chunk, weight)  # [chunk_size, V]

            # Compute softmax and loss
            max_logits = torch.max(logits, dim=-1, keepdim=True)[0]
            exp_logits = torch.exp(logits - max_logits)
            sum_exp = torch.sum(exp_logits, dim=-1, keepdim=True)
            probs = exp_logits / sum_exp  # [chunk_size, V]

            # Compute loss using gather
            correct_logits = torch.gather(
                logits, 1, labels_chunk.unsqueeze(1)
            )  # [chunk_size, 1]
            out_loss += torch.sum(
                max_logits.squeeze()
                + torch.log(sum_exp.squeeze())
                - correct_logits.squeeze()
            )

            # Compute gradients
            dprobs = probs.clone()  # [chunk_size, V]
            dprobs.scatter_(
                1,
                labels_chunk.unsqueeze(1),
                dprobs.gather(1, labels_chunk.unsqueeze(1)) - 1,
            )

            # Accumulate gradients
            weight_grad += dprobs.T @ act_chunk  # [H, V]
            act_grad[b:end_idx] = dprobs @ weight  # [chunk_size, H]

        # Scale gradients
        scale = 1.0 / bs
        weight_grad *= scale
        act_grad *= scale

        ctx.save_for_backward(weight_grad, act_grad)
        return scale * out_loss

    @staticmethod
    def backward(ctx, grad_output):  # type: ignore
        (
            weight_grad,
            act_grad,
        ) = ctx.saved_tensors
        return grad_output * weight_grad, grad_output * act_grad, None


# torch.compile does a good enough job with the kernel here

def cross_entropy(input, target, chunksize = 2048,
                reduction: str = "mean", label_smoothing: float = 0.0, inplace_backward: bool = True):
    return EfficientCrossEntropy.apply(input, target, chunksize,reduction,label_smoothing,inplace_backward)

def fused_cross_entropy(lm_head_weight, act, labels):
    return EfficientCrossEntropyFused.apply(lm_head_weight, act, labels)

if __name__ == "__main__":
    # Test if the forward pass is correct
    ###
    torch.manual_seed(0)
    logits = torch.randn(4, 3, requires_grad=True)
    labels = torch.tensor([0, 1, 2, 1])
    print("Logits:", logits, "Labels:", labels, "exprected gradient:", cross_entropy(logits.detach(), labels))
    loss = F.cross_entropy(logits, labels)
    loss.backward()
    print("Loss:", loss.item(), "Grad:", logits.grad)
    logits.grad.zero_()
    loss = cross_entropy(logits, labels)
    loss.backward()
    print("Loss:", loss.item(), "Grad:", logits.grad)
    
    print("######")
    logits = torch.randn(4, 2, 3, requires_grad=True)
    labels = torch.randn(4, 2, 3)
    logits.grad.zero_()
    loss = -torch.sum(labels * F.log_softmax(logits, dim=-1), dim=-1)
    loss.mean().backward()
    print("Loss:", loss, "Grad:", logits.grad)
    logits.grad.zero_()
    loss = cross_entropy(logits, labels, reduction="none")
    loss.mean().backward()
    print("Loss:", loss, "Grad:", logits.grad)
    