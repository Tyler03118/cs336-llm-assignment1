import torch
import math
from typing import Optional, Callable

# 1. 定义讲义中的 SGD 类
class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"] 
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p] 
                t = state.get("t", 0) 
                grad = p.grad.data 
                # 更新公式: theta_{t+1} = theta_t - (alpha / sqrt(t+1)) * grad
                p.data -= lr / math.sqrt(t + 1) * grad 
                state["t"] = t + 1 
        return loss

# 2. 实验脚本
learning_rates = [1e1, 1e2, 1e3]

for lr in learning_rates:
    print(f"\n--- Testing Learning Rate: {lr} ---")
    # 初始化参数，固定随机种子以便对比
    torch.manual_seed(42)
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    opt = SGD([weights], lr=lr)
    
    for t in range(10): # 运行 10 次迭代
        opt.zero_grad()
        loss = (weights**2).mean() # 目标函数是最小化 w^2
        print(f"Iteration {t}, Loss: {loss.item():.4e}")
        loss.backward()
        opt.step()