from torch import Tensor, nn, triu, ones, inf, softmax
import torch.types

class CausalAttention(nn.Module):
    def __init__(
        self,
        d_in: int, 
        d_out: int, 
        context_length:int, 
        dropout:int = 0.5, 
        qkv_bias:bool = False):
        super().__init__()
        
        self.W_query = nn.Linear(d_in, d_out, qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.context_length = context_length
        self.d_in = d_in
        
        self.register_buffer(
            'mask',
            triu(ones(context_length, context_length), diagonal=1)
        )
        
    def forward(self, inputs:Tensor):
        batch_size, context_length, input_dimension = inputs.shape
        if context_length != self.context_length:
            raise ValueError(
                f"""Input context length does not match layer.
                Layer context length = {self.context_length} 
                Input context length = {context_length}"""
            )
        if input_dimension != self.d_in:
            raise ValueError(
                f"""Input embedding dimension does not match layer
                Layer input dimension = {self.d_in}
                Input dimension = {input_dimension}"""
            )
        
        # QKV are all of size [batch, context, d_out]
        queries:Tensor = self.W_query(inputs)
        keys:Tensor = self.W_key(inputs)
        values:Tensor = self.W_value(inputs)
        
        # Calculate attention score
        # queries [batch, context, d_out]
        # keys.mT [batch, d_out, context]
        # attnscore [batch, context, context]
        attn_scores = queries @ keys.mT
        
        # mask scores for causal
        attn_scores.masked_fill_(self.mask.bool(), -inf)
        
        # softmax normalize
        attn_weights = softmax(attn_scores, dim=-1)
        
        # dropout
        attn_weights = self.dropout(attn_weights)
        
        # calculate context vector
        # attn weights [batch, context, context]
        # values [batch, context, d_out]
        # context vec [batch, context, d_out]
        context_vec = attn_weights @ values
        return context_vec

class MultiheadAttentionWrapper(nn.Module):
    def __init__(
        self,
        d_in:int,
        d_out:int,
        context_length:int,
        num_heads:int,
        dropout:int = 0.5,
        qkv_bias:bool = False,
    ):
        super().__init__()
        self.heads = nn.ModuleList([
            CausalAttention(
                d_in, d_out, context_length, dropout, qkv_bias
            )
            for _ in range(num_heads)
        ])
        
    def forward(self, inputs):
        return torch.cat([head(inputs) for head in self.heads], dim=-1)