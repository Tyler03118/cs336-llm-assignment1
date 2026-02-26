import os
import argparse
import numpy as np
import multiprocessing as mp
from functools import partial
from tqdm import tqdm

from tokenizer import Tokenizer

def process_chunk(chunk, vocab_filepath, merges_filepath):
    """
    Worker function to tokenize a chunk of text.
    We instantiate Tokenizer per process to avoid serialization issues with regex.
    """
    tokenizer = Tokenizer.from_files(
        vocab_filepath, 
        merges_filepath, 
        special_tokens=["<|endoftext|>"]
    )
    return tokenizer.encode(chunk)

def tokenize_file(input_path, output_path, vocab_path, merges_path, num_workers=None):
    if num_workers is None:
        num_workers = max(1, os.cpu_count() - 1)
        
    print(f"Tokenizing {input_path} using {num_workers} workers...")
    
    # Read the file and split by document boundary to avoid cutting tokens
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()
    
    # Split the text by the special token to safely chunk
    delimiter = "<|endoftext|>"
    chunks = text.split(delimiter)
    
    # Re-add the delimiter to all but the last chunk (if we split by it, the delimiter is removed)
    # Actually, TinyStories documents are usually separated by <|endoftext|>.
    # We can just process each document and add the delimiter's ID back.
    
    tokenizer_main = Tokenizer.from_files(vocab_path, merges_path, special_tokens=[delimiter])
    eot_id = tokenizer_main.encode(delimiter)[0]
    
    # Function to apply
    worker_fn = partial(process_chunk, vocab_filepath=vocab_path, merges_filepath=merges_path)
    
    all_tokens = []
    
    with mp.Pool(num_workers) as pool:
        # We use imap or map. tqdm for progress bar
        # We process chunks that are joined docs to reduce overhead, or just map docs
        # Since some docs might be small, let's group them
        batch_size = 1000
        grouped_chunks = [
            delimiter.join(chunks[i:i + batch_size]) + delimiter 
            for i in range(0, len(chunks), batch_size)
        ]
        
        for ids in tqdm(pool.imap(worker_fn, grouped_chunks), total=len(grouped_chunks)):
            all_tokens.extend(ids)
            
    # Save as uint16
    print(f"Total tokens: {len(all_tokens)}")
    np_tokens = np.array(all_tokens, dtype=np.uint16)
    np.save(output_path, np_tokens)
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to input text file")
    parser.add_argument("--output", type=str, required=True, help="Path to output npy file")
    parser.add_argument("--vocab", type=str, default="../tinystories_vocab.json")
    parser.add_argument("--merges", type=str, default="../tinystories_merges.txt")
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    args = parser.parse_args()
    
    tokenize_file(args.input, args.output, args.vocab, args.merges, args.workers)
