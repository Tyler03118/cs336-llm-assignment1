import json
import regex as re
from typing import Iterable, Iterator

# GPT-2 的预分词正则表达式
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        """
        Construct a tokenizer from a given vocabulary, list of merges, and special tokens.
        """
        self.vocab = vocab.copy()
        self.merges = merges
        self.special_tokens = special_tokens if special_tokens is not None else []
        
        # 获取当前词表中最大的 ID，以便为新的 special tokens 分配 ID
        existing_values = list(self.vocab.values())
        next_id = max(self.vocab.keys()) + 1 if self.vocab else 256
        
        # 1. 注入 special tokens
        for st in self.special_tokens:
            st_bytes = st.encode("utf-8")
            if st_bytes not in existing_values:
                self.vocab[next_id] = st_bytes
                next_id += 1
                
        # 2. 构建反向词表 (Bytes -> ID) 方便 encode 时极速查找
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        
        # 3. 构建 merges 优先级字典 (Pair -> Rank)
        # 索引越小，说明合并发生的越早，优先级越高
        self.merge_ranks = {pair: i for i, pair in enumerate(self.merges)}
        
        # 4. 构建 special tokens 的切分正则
        if self.special_tokens:
            escaped = [re.escape(st) for st in self.special_tokens]
            self.special_pattern = re.compile("(" + "|".join(escaped) + ")")
        else:
            self.special_pattern = None

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        """
        Class method that constructs and returns a Tokenizer from serialized files.
        """
        # 完美还原 backslashreplace 的辅助函数
        def unescape(s: str) -> bytes:
            try:
                # 这个操作能把带有 "\\xe2" 的字符串精准还原回 b'\xe2'
                return s.encode('utf-8').decode('unicode_escape').encode('latin-1')
            except Exception:
                return s.encode('utf-8')

        # 读取词表 (JSON)
        with open(vocab_filepath, "r", encoding="utf-8") as f:
            raw_vocab = json.load(f)
            
        vocab = {}
        for k, v in raw_vocab.items():
            k_int = int(k)
            # 基础字节 0-255 强制初始化为单字节，绝对不会出错
            if 0 <= k_int <= 255:
                vocab[k_int] = bytes([k_int])
            else:
                vocab[k_int] = unescape(v) if isinstance(v, str) else v

        # 读取合并规则 (TXT)
        merges = []
        with open(merges_filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip("\n")
                if not line:
                    continue
                parts = line.split(" ")
                if len(parts) == 2:
                    merges.append((unescape(parts[0]), unescape(parts[1])))

        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        """Encode an input text into a sequence of token IDs."""
        if not text:
            return []

        # 1. 首先用特殊 Token 切开文本 (保留特殊 Token 自身)
        if self.special_pattern:
            parts = self.special_pattern.split(text)
        else:
            parts = [text]

        ids = []
        for part in parts:
            if not part:
                continue
                
            # 处理特殊 Token
            if self.special_tokens and part in self.special_tokens:
                ids.append(self.inverse_vocab[part.encode("utf-8")])
                continue

            # 2. 对普通文本进行 GPT-2 正则预分词
            for match in re.finditer(PAT, part):
                token_str = match.group()
                # 初始状态：将字符串转化为基础单字节的列表
                word = [bytes([b]) for b in token_str.encode("utf-8")]

                # 3. 贪心应用 Merges (BPE 核心逻辑)
                while len(word) >= 2:
                    lowest_rank = float('inf')
                    best_idx = -1
                    
                    # 遍历当前 word 中所有的相邻对，寻找在 merges 中排名最靠前的那一对
                    for i in range(len(word) - 1):
                        pair = (word[i], word[i+1])
                        rank = self.merge_ranks.get(pair, float('inf'))
                        if rank < lowest_rank:
                            lowest_rank = rank
                            best_idx = i
                    
                    # 如果找不到任何可以合并的对，说明合并结束，跳出循环
                    if lowest_rank == float('inf'):
                        break
                        
                    # 执行合并：将 word[best_idx] 和 word[best_idx+1] 拼起来
                    new_token = word[best_idx] + word[best_idx+1]
                    word = word[:best_idx] + [new_token] + word[best_idx+2:]

                # 4. 将合并后的最终字节序列转化为对应的 ID
                for w in word:
                    ids.append(self.inverse_vocab[w])

        return ids

    def decode(self, ids: list[int]) -> str:
        """Decode a sequence of token IDs into text."""
        # 根据 ID 找回原字节序列并拼接
        byte_sequence = b"".join(self.vocab[i] for i in ids)
        # [cite_start]使用 errors='replace' 来安全地处理可能出现的非法 Unicode 字节 (将其替换为 U+FFFD) [cite: 1]
        return byte_sequence.decode("utf-8", errors="replace")

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Given an iterable of strings, return a generator that lazily yields token IDs.
        """
        # 由于我们用 GPT-2 正则和 特殊 Token 进行安全切分
        # 每一块 chunk (比如文件的一行) 通常可以直接独立 encode 而不会截断中间的单词
        for chunk in iterable:
            # yield from 可以在不建立巨大中间列表的情况下，一个个地吐出 ID，极其节省内存
            yield from self.encode(chunk)