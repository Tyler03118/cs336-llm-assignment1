import time
from cs336_basics.tokenizer import Tokenizer

def load_sample_docs(filepath, num_docs=10):
    """读取前 N 个文档"""
    with open(filepath, "r", encoding="utf-8") as f:
        # 读取足够多的字符以确保能提取出 10 个文档
        content = f.read(100000) 
    docs = [d for d in content.split("<|endoftext|>") if d.strip()]
    return docs[:num_docs]

def calc_compression(tokenizer, docs, tokenizer_name, data_name):
    """计算压缩比: 原始字节数 / Token 数量"""
    total_bytes = 0
    total_tokens = 0
    for doc in docs:
        total_bytes += len(doc.encode("utf-8"))
        total_tokens += len(tokenizer.encode(doc))
    
    ratio = total_bytes / total_tokens if total_tokens > 0 else 0
    print(f"[{tokenizer_name} Tokenizer] 编码 [{data_name} 数据] -> 压缩比: {ratio:.2f} bytes/token")
    return ratio

def main():
    print("⏳ 正在加载两个 Tokenizer...\n")
    # 加载 TinyStories (10K)
    ts_tokenizer = Tokenizer.from_files(
        "../tinystories_vocab.json", "../tinystories_merges.txt", special_tokens=["<|endoftext|>"]
    )
    # 加载 OWT (32K) - 请确保文件名和你保存的一致
    owt_tokenizer = Tokenizer.from_files(
        "../owt_valid_vocab.json", "../owt_valid_merges.txt", special_tokens=["<|endoftext|>"]
    )

    print("📖 正在加载测试样本...")
    # 请替换为你的实际文件路径
    ts_docs = load_sample_docs("../data/TinyStoriesV2-GPT4-valid.txt", 10) 
    owt_docs = load_sample_docs("../data/owt_valid.txt", 10)

    print("\n📊 --- Deliverable (a): 各自领域的压缩比 ---")
    calc_compression(ts_tokenizer, ts_docs, "TinyStories", "TinyStories")
    calc_compression(owt_tokenizer, owt_docs, "OpenWebText", "OpenWebText")

    print("\n📉 --- Deliverable (b): 跨领域分词测试 ---")
    calc_compression(ts_tokenizer, owt_docs, "TinyStories", "OpenWebText")

    print("\n🚀 --- Deliverable (c): 吞吐量与 The Pile 耗时估算 ---")
    # 使用一段较长的文本测试 OWT Tokenizer 的纯编码速度
    test_text = "".join(owt_docs) * 5
    text_bytes = len(test_text.encode("utf-8"))
    
    start_time = time.time()
    _ = owt_tokenizer.encode(test_text)
    elapsed = time.time() - start_time
    
    throughput = text_bytes / elapsed
    print(f"分词吞吐量: {throughput:,.2f} bytes/second")
    
    # 计算 825GB 需要的时间
    pile_bytes = 825 * 1024**3
    pile_hours = (pile_bytes / throughput) / 3600
    print(f"估算处理 The Pile (825GB) 耗时: {pile_hours:,.2f} 小时")

if __name__ == "__main__":
    main()