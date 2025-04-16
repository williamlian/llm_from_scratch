from torch import Tensor, nn, triu, ones, inf, softmax
import torch.types

class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_in: int, 
        d_out: int, 
        context_length:int,
        num_heads:int, 
        dropout:float = 0.5, 
        qkv_bias:bool = False,
    ):
        super().__init__()
        assert (d_out % num_heads == 0), "d_out must be divisible by num_heads"
        
        self.d_in = d_in
        self.d_out = d_out
        self.context_length = context_length
        
        # dimension per head
        self.num_heads = num_heads
        self.head_dim = self.d_out // num_heads
        
        self.W_query = nn.Linear(d_in, d_out, qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, qkv_bias)
        self.dropout = nn.Dropout(dropout)
        # Linear layer to project concatenated multi-head outputs
        self.out_proj = nn.Linear(d_out, d_out)
        
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
        
        ######################################################################################### 
        # MULTI HEAD SPLIT
        #########################################################################################
        # Split d_out into multiple dimensions, i.e. num_heads x dim_head
        # after the split, the dimensions now are
        # [batch, context, num_heads, head_dim]
        queries_multi = queries.view(batch_size, context_length, self.num_heads, self.head_dim)
        keys_multi = keys.view(batch_size, context_length, self.num_heads, self.head_dim)
        values_multi = values.view(batch_size, context_length, self.num_heads, self.head_dim)
        # Now we switch the dimensions so the last two dims are [context x dim_head] for later self correlation
        queries_multi.transpose_(1, 2)
        keys_multi.transpose_(1, 2)
        values_multi.transpose_(1, 2)
        
        ######################################################################################### 
        # CALCULATE ATTENTION SCORE (Q * K)
        #########################################################################################
        #
        # Now QKV are of dimension [batch, num_heads, context, dim_head]
        #
        # Calculate attention score as before
        # SIZE = [batch, num_heads, context, context]
        attn_scores = queries_multi @ keys_multi.mT
        # mask scores for causal
        attn_scores.masked_fill_(self.mask.bool(), -inf) # type: ignore
        # softmax normalize
        attn_weights = softmax(attn_scores / self.head_dim**0.5, dim=-1)
        # dropout
        attn_weights:Tensor = self.dropout(attn_weights)
        
        ######################################################################################### 
        # CALCULATE CONTEXT VECTOR (ATTN * V)
        #########################################################################################
        # attn weights [batch, num_heads, context, context]
        # values [batch, num_head, context, head_dim]
        # context vec [batch, num_head, context, head_dim]
        context_vec = attn_weights @ values_multi
        
        ######################################################################################### 
        # MULTI HEAD COMBINE, RESULT SIZE [BATCH, CONTEXT, D_OUT]
        #########################################################################################
        # context vec size = [batch, context, num_head, head_dim]
        context_vec.transpose_(1, 2)
        # Merge last two dimensions so we have [context, d_out] per batch
        context_vec = context_vec.contiguous().view(batch_size, context_length, self.d_out)
        
        ######################################################################################### 
        # FINAL PROJECTION
        ########################################################################################
        context_vec = self.out_proj(context_vec)
        
        return context_vec

