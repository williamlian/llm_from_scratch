import torch
import torch.nn as nn

class SelfAttentionV2(nn.Module):
    def __init__(self, dim_in:int, dim_out:int, qkv_bias:bool=False):
        super().__init__()
        self.W_query = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.W_key = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.W_value = nn.Linear(dim_in, dim_out, bias=qkv_bias)
    
    def forward(self, x:torch.tensor) -> torch.tensor:
        # x in shape of [context_size, embed dimensions]
        queries = self.W_query(x) # [context * dim out]
        keys = self.W_key(x)
        values = self.W_value(x)
         
        # rotate key, so each column is a key for a token
        # query(i) dot key(j) is the attemtion for pair(i,j)
        # [context * dim_out] * [dim_out * context]
        # = [context * context]
        attention_score = queries @ keys.T
        attention_weights = torch.softmax(
            attention_score / keys.shape[-1] ** 0.5,
            dim=-1
        )
        
        # context vec: for each row in value
        # dot product of all input and the attention 
        # weight (i)
        # aW => [context * context]
        # values => [contest * dim_out]
        context_vec = attention_weights @ values
        return context_vec
        