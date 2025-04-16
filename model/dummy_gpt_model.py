import torch
import torch.nn as nn
from torch.types import Tensor

from config import GPTConfig

class DummyGPTModel(nn.Module):
    def __init__(self, cfg:GPTConfig):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg['vocab_size'], cfg['emb_dim'])
        self.pos_emb = nn.Embedding(cfg['context_length'], cfg['emb_dim'])
        self.drop_emb = nn.Dropout(cfg['drop_rate'])
        self.trf_blocks = nn.Sequential(
            *[DummyTransformerBlock(cfg)
              for _ in range(cfg['n_layers'])]
        )
        self.final_norm = DummyLayerNorm(cfg['emb_dim'])
        self.out_head = nn.Linear(cfg['emb_dim'], cfg['vocab_size'], bias=False)
    

class DummyTransformerBlock(nn.Module):
    def __init__(self, cfg:GPTConfig):
        super().__init__()
        
    def forward(self, x:Tensor):
        return x

class DummyLayerNorm(nn.Module):
    def __init__(self, normalized_shape:int, eps=1e-5):
        super().__init__()
    
    def forward(self, x:Tensor):
        return x
