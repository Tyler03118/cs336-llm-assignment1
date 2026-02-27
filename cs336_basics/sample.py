import torch
import torch.nn.functional as F
from model import TransformerLM
from tokenizer import Tokenizer
from checkpoint import load_checkpoint, find_latest_checkpoint

def top_p_sampling(logits, top_p=0.9):
    """核采样 (Nucleus Sampling) 逻辑"""
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    
    # 移除累积概率超过 top_p 的 token
    sorted_indices_to_remove = cumulative_probs > top_p
    # 将第一个超过 top_p 的 token 保留（所以往右移一位）
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    
    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
    logits[indices_to_remove] = float('-inf')
    return logits

@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=1.0, top_p=0.9):
    """自回归生成循环"""
    model.eval()
    for _ in range(max_new_tokens):
        # 如果长度超过训练时的 context_length，进行裁剪
        idx_cond = idx if idx.size(1) <= 256 else idx[:, -256:]
        
        logits = model(idx_cond)
        # 只取序列中最后一个位置的预测结果
        # Temperature 缩放
        logits = logits[:, -1, :] / temperature
        
        if top_p < 1.0:
            logits = top_p_sampling(logits, top_p=top_p)
            
        probs = F.softmax(logits, dim=-1)

        # 根据概率分布进行随机采样
        idx_next = torch.multinomial(probs, num_samples=1)
        
        # 将刚才采样得到的 idx_next（新词的 ID）拼接在现有序列 idx 的末尾
        idx = torch.cat((idx, idx_next), dim=1)
        
        # 如果生成了 <|endoftext|> 则提前停止
        if idx_next.item() == 0: # 假设 0 是你的 EOT ID
            break
    return idx

# --- 执行加载与生成 ---
device = 'mps' # 你的 Mac 环境
# 使用 TinyStories 的分词器文件
tokenizer = Tokenizer.from_files("tinystories_vocab.json", "tinystories_merges.txt", special_tokens=["<|endoftext|>"])
eot_id = tokenizer.encode("<|endoftext|>")[0]

# 核心修复：使用 checkpoint 工具函数加载
checkpoint_path = find_latest_checkpoint("out")
if not checkpoint_path:
    raise FileNotFoundError("No checkpoints found in 'out/' directory.")

print(f"Loading checkpoint from {checkpoint_path}...")

# 必须与 train.py 中的参数完全一致
model = TransformerLM(
    vocab_size=10000, 
    d_model=512, 
    num_layers=4, 
    num_heads=16, 
    context_length=256, 
    d_ff=1344, 
    theta=10000.0
)

# 使用 checkpoint.py 中的 load_checkpoint，它已经处理了 _orig_mod. 前缀
load_checkpoint(checkpoint_path, model)
model.to(device)

def generate_with_eot(model, idx, max_new_tokens, temperature=1.0, top_p=0.9, eot_id=None):
    """带 EOT 停止逻辑的生成"""
    model.eval()
    for _ in range(max_new_tokens):
        idx_cond = idx if idx.size(1) <= 256 else idx[:, -256:]
        logits = model(idx_cond)
        logits = logits[:, -1, :] / temperature
        if top_p < 1.0:
            logits = top_p_sampling(logits, top_p=top_p)
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        next_token_str = tokenizer.decode([idx_next.item()])
        print(next_token_str, end='', flush=True)
        idx = torch.cat((idx, idx_next), dim=1)
        if eot_id is not None and idx_next.item() == eot_id:
            break
    return idx

prompt = "In 2026, there was a little boy in Georgia."
input_ids = torch.tensor([tokenizer.encode(prompt)], device=device)

print(f"\n--- Generating for prompt: \"{prompt}\" ---")
output_ids = generate_with_eot(model, input_ids, max_new_tokens=256, temperature=0.8, top_p=0.9, eot_id=eot_id)
# print(tokenizer.decode(output_ids[0].tolist())) # Using stream output now