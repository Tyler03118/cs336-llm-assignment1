import os
import argparse
import time
import numpy as np
import torch
from dotenv import load_dotenv
import wandb

from model import TransformerLM, cross_entropy
from optimizer import AdamW, clip_gradients, get_lr_cosine_schedule
from data import get_batch
from checkpoint import save_checkpoint

# 加载 .env 中的 WANDB_API_KEY
load_dotenv()

# -----------------------------------------------------------------------------
# 1. 评估辅助函数：定期在 Train 和 Val 集上计算真实的平均 Loss
# -----------------------------------------------------------------------------
@torch.no_grad()
def estimate_loss(model, train_data, val_data, eval_iters, batch_size, context_length, device):
    """在不计算梯度的情况下，评估模型当前性能"""
    out = {}
    model.eval() # 切换到评估模式
    for split, data in [('train', train_data), ('val', val_data)]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(data, batch_size, context_length, device)
            logits = model(X)
            
            # 错位对齐计算 Loss
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = Y[:, :-1].contiguous()
            loss = cross_entropy(shift_logits, shift_labels)
            
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train() # 切回训练模式
    return out

# -----------------------------------------------------------------------------
# 2. 主训练逻辑
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train a Transformer Language Model (TinyStories)")
    # 数据与设备参数
    parser.add_argument('--train_data', type=str, required=True, help="Path to train.npy")
    parser.add_argument('--val_data', type=str, required=True, help="Path to val.npy")
    parser.add_argument('--out_dir', type=str, default='out', help="Directory to save checkpoints")
    parser.add_argument('--device', type=str, default='mps', help="Device to train on (cpu, cuda, mps)")
    
    # === 讲义 7.2 规定的 17M 模型超参数 ===
    parser.add_argument('--vocab_size', type=int, default=10000)
    parser.add_argument('--context_length', type=int, default=256)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--num_heads', type=int, default=16)
    parser.add_argument('--d_ff', type=int, default=1344)
    parser.add_argument('--theta', type=float, default=10000.0)
    
    # === 低资源极速通关版训练超参数 (40M Tokens) ===
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_iters', type=int, default=5000)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--min_lr', type=float, default=1e-4)
    parser.add_argument('--warmup_iters', type=int, default=100)
    parser.add_argument('--lr_decay_iters', type=int, default=5000) # 严格匹配 max_iters
    parser.add_argument('--weight_decay', type=float, default=0.1)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    
    # 日志与保存
    parser.add_argument('--eval_interval', type=int, default=250)
    parser.add_argument('--eval_iters', type=int, default=20)
    
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # 1. 初始化 W&B 监控
    wandb.init(
        project="cs336-tinystories", 
        entity="zijili-georgia-institute-of-technology", 
        name=f"17M_fast_lr{args.learning_rate}", 
        config=vars(args)
    )

    # 2. 内存映射方式加载数据
    train_data = np.memmap(args.train_data, dtype=np.uint16, mode='r')
    val_data = np.memmap(args.val_data, dtype=np.uint16, mode='r')
    print(f"Loaded train data: {len(train_data)} tokens")

    # 3. 初始化模型并移至设备
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        theta=args.theta
    )
    model.to(args.device)

    # === Apple Silicon 专属加速优化 (讲义要求) ===
    if args.device == 'mps':
        print("Enabling torch.compile with aot_eager for MPS...")
        model = torch.compile(model, backend="aot_eager")
    elif args.device == 'cpu':
        print("Enabling torch.compile for CPU...")
        model = torch.compile(model)
    # 绝对不能在这里写 TF32 的优化逻辑，防止内核损坏！

    # 4. 初始化优化器
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # 5. 训练大循环
    X, Y = get_batch(train_data, args.batch_size, args.context_length, args.device)
    training_start_time = time.time()
    t0 = time.time()

    print(f"--- Starting Training on {args.device} for {args.max_iters} iters ---")
    for iter_num in range(1, args.max_iters + 1):
        # A. 动态设置当前步的学习率
        lr = get_lr_cosine_schedule(iter_num, args.learning_rate, args.min_lr, args.warmup_iters, args.lr_decay_iters)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # B. 前向传播与计算 Loss
        logits = model(X)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = Y[:, :-1].contiguous()
        loss = cross_entropy(shift_logits, shift_labels)

        # C. 反向传播
        optimizer.zero_grad()
        loss.backward()

        # D. 梯度裁剪
        clip_gradients(model.parameters(), max_norm=args.grad_clip)

        # E. 更新权重
        optimizer.step()

        # F. 预加载下一个 Batch
        X, Y = get_batch(train_data, args.batch_size, args.context_length, args.device)

        # G. 定期评估并保存 Checkpoint
        if iter_num % args.eval_interval == 0 or iter_num == args.max_iters:
            losses = estimate_loss(model, train_data, val_data, args.eval_iters, args.batch_size, args.context_length, args.device)
            total_elapsed_time = time.time() - training_start_time
            dt = time.time() - t0
            
            # W&B 自动画图
            wandb.log({
                "step": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "wallclock_time": total_elapsed_time,
            })

            print(f"Iter {iter_num:4d} | Train Loss {losses['train']:.4f} | Val Loss {losses['val']:.4f} | LR {lr:.4e} | Time {total_elapsed_time:.2f}s")
            
            ckpt_path = os.path.join(args.out_dir, f"ckpt_iter{iter_num}.pt")
            save_checkpoint(model, optimizer, iter_num, ckpt_path)
            t0 = time.time() 

    wandb.finish()
    print("Training Complete! Valid loss should be around or below 2.00.")

if __name__ == '__main__':
    main()