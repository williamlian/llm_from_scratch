
from typing import TypedDict


class GPTConfig(TypedDict, total=True):
    vocab_size: int
    context_length: int
    emb_dim: int
    n_heads: int
    n_layers: int
    drop_rate: float
    qkv_bias: bool

GPT_CONFIG_124M:GPTConfig = {
    "vocab_size": 50257,    # Vocabulary size - match BPE tokenizer
    "context_length": 1024, # context window size
    "emb_dim": 768,         # embedding dimension, i.e. d_in
    "n_heads": 12,          # Number of attention heads
    "n_layers": 12,         # Number of layers
    "drop_rate": 0.1,       # Dropout rate
    "qkv_bias": False,      # QKV bias
}