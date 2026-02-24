import torch
import math
from typing import Optional, Callable

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        # 1. 基础参数校验
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        # 将超参数存入 defaults 字典，传给基类
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            # 获取当前参数组的超参数
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                state = self.state[p]

                # 2. 初始化状态 (m=0, v=0, t=0)
                if len(state) == 0:
                    state["t"] = 0
                    # 使用 zeros_like 确保 m 和 v 的形状和设备与参数 p 完全一致
                    state["m"] = torch.zeros_like(p.data)
                    state["v"] = torch.zeros_like(p.data)

                m, v = state["m"], state["v"]
                
                # 更新步数 t
                state["t"] += 1
                t = state["t"]

                # 3. 更新一阶矩 m: m = beta1 * m + (1 - beta1) * g
                # 使用 in-place 乘法和加法节省显存
                m.mul_(beta1).add_(grad, alpha=1 - beta1)

                # 4. 更新二阶矩 v: v = beta2 * v + (1 - beta2) * g^2
                # addcmul_ 是高效的逐元素乘加操作：v = v + value * (grad * grad)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # 5. 计算偏差修正后的学习率 alpha_t
                # alpha_t = lr * sqrt(1 - beta2^t) / (1 - beta1^t)
                bias_correction2 = math.sqrt(1 - beta2 ** t)
                bias_correction1 = 1 - beta1 ** t
                alpha_t = lr * bias_correction2 / bias_correction1

                # 6. 参数更新: theta = theta - alpha_t * (m / (sqrt(v) + eps))
                # 使用 addcdiv_ 进行高效除法并累加
                denom = v.sqrt().add_(eps)
                p.data.addcdiv_(m, denom, value=-alpha_t)

                # 7. 应用权重衰减: theta = theta - lr * lambda * theta
                # 注意：根据算法1，这里的衰减使用的是基础 lr，而不是 alpha_t
                if weight_decay > 0.0:
                    p.data.add_(p.data, alpha=-lr * weight_decay)

        return loss

import math

def get_lr_cosine_schedule(t: int, alpha_max: float, alpha_min: float, Tw: int, Tc: int) -> float:
    """
    计算带有预热的余弦退火学习率。
    
    参数:
        t: 当前迭代步数
        alpha_max: 最大学习率 (预热结束时的值)
        alpha_min: 最小学习率 (退火结束后的值)
        Tw: 预热 (Warm-up) 的总步数
        Tc: 余弦退火 (Cosine Annealing) 结束的总步数
    """
    # 1. 线性预热阶段 (Warm-up)
    if t < Tw:
        return (t / Tw) * alpha_max
        
    # 2. 余弦退火阶段 (Cosine annealing)
    elif Tw <= t <= Tc:
        # 对应公式: alpha_min + 0.5 * (1 + cos((t - Tw) / (Tc - Tw) * pi)) * (alpha_max - alpha_min)
        progress = (t - Tw) / (Tc - Tw)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return alpha_min + cosine_decay * (alpha_max - alpha_min)
        
    # 3. 平缓微调阶段 (Post-annealing)
    else:
        return alpha_min


def clip_gradients(parameters, max_norm: float, eps: float = 1e-6):
    """
    对模型参数的梯度进行全局 L2 范数裁剪。
    
    参数:
        parameters: 模型的参数列表 (通常是 model.parameters())
        max_norm: 允许的最大全局 L2 范数 (M)
        eps: 用于数值稳定的极小值
    """
    # 1. 过滤掉没有梯度的参数
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return
    
    # 2. 计算全局 L2 范数的平方和
    # 计算每个梯度的平方和，然后全部加起来
    total_norm_sq = torch.tensor(0.0, device=grads[0].device)
    for g in grads:
        total_norm_sq += torch.sum(g.detach() ** 2)
        
    # 3. 开根号得到真实的全局 L2 范数
    total_norm = torch.sqrt(total_norm_sq)
    
    # 4. 判断并就地缩放 (In-place scaling)
    if total_norm > max_norm:
        scale_factor = max_norm / (total_norm + eps)
        for g in grads:
            g.detach().mul_(scale_factor)
