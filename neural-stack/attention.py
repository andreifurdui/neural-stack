import torch
import torch.nn as nn
from typing import Optional


class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention mechanism.

    Computes attention weights using the scaled dot-product between queries and keys,
    then applies these weights to values.
    """

    def __init__(self, dim_model: int, dropout: float = 0.1) -> None:
        """Initialize the attention mechanism.

        Args:
            dim_model: Dimension of the key/query/value vectors per head.
            dropout: Dropout probability applied to attention weights. Default: 0.1.
        """
        super(ScaledDotProductAttention, self).__init__()

        self.scaling_factor = torch.sqrt(dim_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the scaled dot-product attention.

        Args:
            query: Tensor of shape [B, num_heads, T_q, dim_kv]
            key: Tensor of shape [B, num_heads, T_kv, dim_kv]
            value: Tensor of shape [B, num_heads, T_kv, dim_kv]
            mask: Optional mask tensor. Positions with 0 are masked out. Default: None.

        Returns:
            Tuple of (output, attention_scores):
                - output: Attention output of shape [B, num_heads, T_q, dim_kv]
                - attention_scores: Attention weights of shape [B, num_heads, T_q, T_kv]
        """
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / self.scaling_factor # [B, num_heads, T_q, T_kv]

        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)

        attention_scores = torch.softmax(attention_scores, dim=-1) # [B, num_heads, T_q, T_kv]
        attention_scores = self.dropout(attention_scores)          # [B, num_heads, T_q, T_kv]

        output = torch.matmul(attention_scores, value)             # [B, num_heads, T_q, dim_kv]

        return output, attention_scores                            # [B, num_heads, T_q, dim_kv], [B, num_heads, T_q, T_kv]
    
class MultiHeadAttention(nn.Module):
    """Multi-Head Attention mechanism.

    Projects queries, keys, and values to multiple heads, applies scaled dot-product
    attention in parallel, and combines the results with a final linear projection.
    """

    def __init__(self, num_heads: int, dim_model: int, dropout: float = 0.1) -> None:
        """Initialize the multi-head attention layer.

        Args:
            num_heads: Number of parallel attention heads.
            dim_model: Model dimension. Must be divisible by num_heads.
            dropout: Dropout probability applied to attention weights. Default: 0.1.
        """
        super(MultiHeadAttention, self).__init__()

        assert dim_model % num_heads == 0, "dim_model must be divisible by num_heads"

        self.num_heads = num_heads
        self.dim_kv = dim_model // num_heads

        self.q_proj = nn.Linear(dim_model, dim_model)
        self.k_proj = nn.Linear(dim_model, dim_model)
        self.v_proj = nn.Linear(dim_model, dim_model)
        self.out_proj = nn.Linear(dim_model, dim_model)

        self.scaled_dot_product_attention = ScaledDotProductAttention(self.dim_kv, dropout)

    def forward(self, key: torch.Tensor, query: torch.Tensor, value: torch.Tensor, mask: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply multi-head attention.

        Args:
            key: Key tensor of shape [B, T_kv, dim_model]
            query: Query tensor of shape [B, T_q, dim_model]
            value: Value tensor of shape [B, T_kv, dim_model]
            mask: Optional mask tensor. Positions with 0 are masked out. Default: None.

        Returns:
            Tuple of (output, attention_scores):
                - output: Attention output of shape [B, T_q, dim_model]
                - attention_scores: Attention weights of shape [B, num_heads, T_q, T_kv]
        """
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