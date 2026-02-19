import time
import json
import os
from cs336_basics.train_bpe import train_bpe

def save_tokenizer(vocab, merges, prefix):
    """Serialize vocabulary and merges to disk."""
    # 保存 Vocab (将 bytes 转换为可以被 JSON 序列化的 ISO-8859-1 字符串，或者直接存 hex)
    # 为了方便肉眼检查，这里我们将 bytes 解码（遇到无法解码的忽略或替换）
    readable_vocab = {}
    for vid, token_bytes in vocab.items():
        # 用 backslashreplace 保证所有字节都能存下来
        readable_vocab[vid] = token_bytes.decode('utf-8', errors='backslashreplace')
        
    with open(f"{prefix}_vocab.json", "w", encoding="utf-8") as f:
        json.dump(readable_vocab, f, indent=2, ensure_ascii=False)
        
    # 保存 Merges
    with open(f"{prefix}_merges.txt", "w", encoding="utf-8") as f:
        for t1, t2 in merges:
            s1 = t1.decode('utf-8', errors='backslashreplace')
            s2 = t2.decode('utf-8', errors='backslashreplace')
            f.write(f"{s1} {s2}\n")
    print(f"Saved to {prefix}_vocab.json and {prefix}_merges.txt")

def run_experiment(dataset_path, vocab_size, name):
    print(f"--- Starting BPE Training on {name} ---")
    start_time = time.time()
    
    # 核心训练调用
    vocab, merges = train_bpe(
        input_path=dataset_path,
        vocab_size=vocab_size,
        special_tokens=["<|endoftext|>"]
    )
    
    elapsed_time = time.time() - start_time
    print(f"Training Time: {elapsed_time / 60:.2f} minutes")
    
    # 找出最长的 Token
    longest_token_bytes = max(vocab.values(), key=len)
    longest_token_str = longest_token_bytes.decode('utf-8', errors='replace')
    print(f"Longest Token (length {len(longest_token_bytes)}): {repr(longest_token_str)}")
    
    # 序列化到磁盘
    save_tokenizer(vocab, merges, name)
    print("-" * 40)

if __name__ == "__main__":
    # 1. 运行 TinyStories 实验 (10K vocab)
    run_experiment(
        dataset_path="data/TinyStoriesV2-GPT4-train.txt", # 替换为你的实际路径
        vocab_size=10000,
        name="tinystories"
    )
    
    # 注意：OpenWebText 的训练时间会非常长，建议先跑完 TinyStories 再跑 OWT
    # run_experiment(
    #     dataset_path="data/OpenWebText.txt", # 替换为你的实际路径
    #     vocab_size=32000,
    #     name="openwebtext"
    # )