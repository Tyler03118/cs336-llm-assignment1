import os
import regex as re
import multiprocessing
from collections import Counter, defaultdict
from typing import BinaryIO

# GPT-2 的预分词正则表达式
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)
        while True:
            mini_chunk = file.read(mini_chunk_size)

            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    return sorted(set(chunk_boundaries))

def _worker_process(args):
    """
    多进程 Worker：读取文件块，处理特殊 token，使用正则分词，并统计词频。
    """
    input_path, start, end, special_tokens = args
    
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk_bytes = f.read(end - start)
        
    text = chunk_bytes.decode("utf-8", errors="ignore")
    
    # 隔离特殊 token：使用 re.split 切开文本 [cite: 213, 214, 216]
    if special_tokens:
        escaped_tokens = [re.escape(st) for st in special_tokens]
        split_pattern = "(" + "|".join(escaped_tokens) + ")"
        parts = re.split(split_pattern, text)
    else:
        parts = [text]
        
    local_counts = Counter()
    for part in parts:
        # 忽略空字符串和特殊 token [cite: 215]
        if not part or part in special_tokens:
            continue
            
        # 运行 GPT-2 正则查找 pre-tokens [cite: 153]
        for match in re.finditer(PAT, part):
            token_str = match.group()
            # 将切分后的 string 转为 byte tuple，确保每个元素是一个独立的 bytes 对象
            token_bytes = tuple(bytes([b]) for b in token_str.encode("utf-8"))
            local_counts[token_bytes] += 1
            
    return local_counts

def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    # 1. 初始化 Vocab 和特殊 Token [cite: 141, 142]
    vocab = {i: bytes([i]) for i in range(256)} 
    merges = []
    
    next_id = 256
    for st in special_tokens:
        vocab[next_id] = st.encode('utf-8')
        next_id += 1
        
    num_merges = vocab_size - len(vocab)
    if num_merges <= 0:
        return vocab, merges

    # 2. 多进程预分词 (Pre-tokenization) [cite: 205, 206]
    # 使用所有可用核心，稍微预留一点资源给系统
    num_processes = max(1, os.cpu_count() - 1)
    
    with open(input_path, "rb") as f:
        # 默认使用第一个 special_token 作为切分界限，如果没有则默认用 <|endoftext|>
        split_token = special_tokens[0].encode("utf-8") if special_tokens else b"<|endoftext|>"
        boundaries = find_chunk_boundaries(f, num_processes, split_token)
        
    chunk_args = [(input_path, boundaries[i], boundaries[i+1], special_tokens) 
                  for i in range(len(boundaries)-1)]
                  
    word_counts = Counter()
    # 启动进程池，收集各个 chunk 的计数结果
    with multiprocessing.Pool(processes=num_processes) as pool:
        for res in pool.imap_unordered(_worker_process, chunk_args):
            word_counts.update(res)

    # 3. 统计初始的相邻字节对频率 [cite: 165]
    pair_counts = defaultdict(int)
    for word, count in word_counts.items():
        for i in range(len(word) - 1):
            pair_counts[(word[i], word[i+1])] += count

    # 4. BPE 核心合并循环 [cite: 165, 166, 167]
    for _ in range(num_merges):
        # 清理由于增量更新导致频率变为 0 或负数的键
        # 这一步非常重要，能防止最大值函数意外选中无效的对
        keys_to_delete = [k for k, v in pair_counts.items() if v <= 0]
        for k in keys_to_delete:
            del pair_counts[k]
            
        if not pair_counts:
            break
            
        # a. 找到频率最高的 pair，规则要求 frequency 相同时优先选字典序更大的 [cite: 169]
        best_pair = max(pair_counts.items(), key=lambda x: (x[1], x[0]))[0]
        
        # b. 记录 merge 和更新 vocab [cite: 166, 167]
        merges.append(best_pair)
        new_token = best_pair[0] + best_pair[1]
        vocab[next_id] = new_token
        next_id += 1
        
        # c. 增量更新 (Incremental Update) [cite: 219, 220]
        new_word_counts = Counter()
        for word, count in word_counts.items():
            # 快速前置检查：如果最佳对的第一个元素都不在词里，说明绝对没必要去扫描
            if best_pair[0] not in word:
                new_word_counts[word] += count
                continue
                
            pair_exists = False
            for i in range(len(word) - 1):
                if word[i] == best_pair[0] and word[i+1] == best_pair[1]:
                    pair_exists = True
                    break
            
            if pair_exists:
                # 减去旧的相邻对频率
                for i in range(len(word) - 1):
                    pair_counts[(word[i], word[i+1])] -= count
                
                # 从左到右贪心合并，生成新词
                new_word = []
                i = 0
                while i < len(word):
                    if i < len(word) - 1 and word[i] == best_pair[0] and word[i+1] == best_pair[1]:
                        new_word.append(best_pair[0] + best_pair[1])
                        i += 2 
                    else:
                        new_word.append(word[i])
                        i += 1
                new_word = tuple(new_word)
                
                # 加上新生成的相邻对频率
                for i in range(len(new_word) - 1):
                    pair_counts[(new_word[i], new_word[i+1])] += count
                
                new_word_counts[new_word] += count
            else:
                new_word_counts[word] += count
                
        word_counts = new_word_counts

    return vocab, merges