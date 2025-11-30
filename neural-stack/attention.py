import torch

import torch.nn as nn


class ScaledDotProductAttention(nn.Module):
    def __init__(self, dim_model, dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()

        self.scaling_factor = torch.sqrt(dim_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask=None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the scaled dot-product attention.
        Args:
            query: Tensor of shape [B, num_heads, T_q, dim_kv]
            key: Tensor of shape [B, num_heads, T_kv, dim_kv]
            value: Tensor of shape [B, num_heads, T_kv, dim_kv]
        Returns:
        """
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / self.scaling_factor # [B, num_heads, T_q, T_kv]

        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)

        attention_scores = torch.softmax(attention_scores, dim=-1) # [B, num_heads, T_q, T_kv]
        attention_scores = self.dropout(attention_scores)          # [B, num_heads, T_q, T_kv]

        output = torch.matmul(attention_scores, value)             # [B, num_heads, T_q, dim_kv]

        return output, attention_scores                            # [B, num_heads, T_q, dim_kv], [B, num_heads, T_q, T_kv]
    
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, dim_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        assert dim_model % num_heads == 0, "dim_model must be divisible by num_heads"
        
        self.num_heads = num_heads
        self.dim_kv = dim_model // num_heads

        self.q_proj = nn.Linear(dim_model, dim_model)
        self.k_proj = nn.Linear(dim_model, dim_model)
        self.v_proj = nn.Linear(dim_model, dim_model)
        self.out_proj = nn.Linear(dim_model, dim_model)

        self.scaled_dot_product_attention = ScaledDotProductAttention(self.dim_kv, dropout)

    def forward(self, key: torch.Tensor, query: torch.Tensor, value: torch.Tensor, mask=None):
        batch_size = query.size(0)

        key = self.k_proj(key)   # [B, T_kv, dim_model]
        query = self.q_proj(query) # [B, T_q, dim_model]
        value = self.v_proj(value) # [B, T_kv, dim_model]

        key = key.view(batch_size, -1, self.num_heads, self.dim_kv).transpose(1, 2) # [B, num_heads, T_kv, dim_model/num_heads]
        query = query.view(batch_size, -1, self.num_heads, self.dim_kv).transpose(1, 2) # [B, num_heads, T_q, dim_model/num_heads]
        value = value.view(batch_size, -1, self.num_heads, self.dim_kv).transpose(1, 2) # [B, num_heads, T_kv, dim_model/num_heads]

        out, attention_scores = self.scaled_dot_product_attention(query, key, value, mask) # [B, num_heads, T_q, dim_model/num_heads], [B, num_heads, T_q, T_kv]

        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.dim_kv) # [B, T_q, dim_model]
        out = self.out_proj(out) # [B, T_q, dim_model]

        return out, attention_scores # [B, T_q, dim_model], [B, num_heads, T_q, T_kv]