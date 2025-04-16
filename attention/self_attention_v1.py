import torch
import torch.nn as nn

class SelfAttentionV1(nn.Module):
    def __init__(self, dim_in:int, dim_out:int):
        super().__init__()
        self.W_query = nn.Parameter(torch.rand(dim_in, dim_out))
        self.W_key = nn.Parameter(torch.rand(dim_in, dim_out))
        self.W_value = nn.Parameter(torch.rand(dim_in, dim_out))
    
    def forward(self, x:torch.tensor) -> torch.tensor:
        # x in shape of [context_size, embed dimensions]
        queries = x @ self.W_query # [context * dim out]
        keys = x @ self.W_key
        values = x @ self.W_value
         
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
        