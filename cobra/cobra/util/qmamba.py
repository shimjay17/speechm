"""
qmamba.py

QMamba submodule definitions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math
from cobra.models.mamba.modeling_mamba import Mamba
from functools import partial

from flash_attn import flash_attn_func as flash_attn


class Attention(nn.Module):
    def __init__(self,
        dim: int,
        num_heads: int = 16,
        bias: bool = True,
        dropout: float = 0.0,
        batch_first: bool = True,
        use_flash: bool = True
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "Embedding dimension must be divisible by number of heads."

        self.use_flash = use_flash
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.inner_dim = dim
        if self.head_dim > 256:
            self.head_dim = 256
            self.inner_dim = 256 * self.num_heads
        self.scale = self.head_dim ** -0.5
        self.dropout_p = dropout


        if use_flash:
            # Define projection layers for query, key, and value
            self.wQ = nn.Linear(dim, self.inner_dim, bias=bias)
            self.wK = nn.Linear(dim, self.inner_dim, bias=bias)
            self.wV = nn.Linear(dim, self.inner_dim, bias=bias)
            self.attn_drop = nn.Dropout(dropout)
            self.proj = nn.Linear(self.inner_dim, dim)
            self.proj_drop = nn.Dropout(dropout)
        else:
            # Using PyTorch's built-in multihead attention
            self.attn = nn.MultiheadAttention(
                dim, 
                num_heads,
                bias=bias,
                dropout=dropout,
                batch_first=batch_first
            )

    def forward(self, x: torch.Tensor, c: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the Attention module.

        Parameters:
        - x: Input tensor (batch_size, seq_len_q, dim)
        - c: Context tensor (batch_size, seq_len_k, dim)

        Returns:
        - out: Output tensor after attention (batch_size, seq_len_q, dim)
        """
        batch_size, seq_len, dim = x.shape

        if c is None:
            c = x   # Self-Attention

        if self.use_flash:
            # Flash Attention Implementation
            Q = self.wQ(x)  # (batch_size, seq_len_q, dim)
            K = self.wK(c)  # (batch_size, seq_len_k, dim)
            V = self.wV(c)  # (batch_size, seq_len_k, dim)

            # Reshape to (batch_size, seq_len, num_heads, head_dim)
            Q = Q.view(Q.size(0), Q.size(1), self.num_heads, self.head_dim)
            K = K.view(K.size(0), K.size(1), self.num_heads, self.head_dim)
            V = V.view(V.size(0), V.size(1), self.num_heads, self.head_dim)

            # Apply flash attention
            out = flash_attn(
                Q, K, V,
                dropout_p=self.dropout_p if self.training else 0.0,
                causal=False
            )   # (batch_size, seq_len_q, num_heads, head_dim)
            
            # Transpose and reshape back to (batch_size, seq_len_q, dim)
            out = out.contiguous().view(batch_size, seq_len, dim)

            # Final linear projection
            out = self.proj(out)
            out = self.proj_drop(out)

            return out
        else:
            # Multihead attention requires (seq_len, batch_size, dim)
            out, _ = self.attn(x, c, c)

            return out


class MambaMixerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        len: Optional[int] = 27 * 27,
        sequence_mix_type: str = "mamba",
        channel_mix_type: str = "mlp", 
        use_cross_attention: bool = True
    ) -> None:
        super().__init__()
        self.sequence_mix_type = sequence_mix_type
        self.channel_mix_type = channel_mix_type

        if sequence_mix_type == "mamba":
            self.sequence_mixer = Mamba(dim)
        elif sequence_mix_type == "self-attention":
            self.sequence_mixer = Attention(dim)
        else:
            raise ValueError(f"Sequence Mixer with `{sequence_mix_type = }` is not supported!")

        self.use_cross_attention = use_cross_attention
        if use_cross_attention:
            self.cross_attention = Attention(dim) 
            self.cross_proj = nn.Linear(2 * dim, dim)

        if channel_mix_type == "mlp":
            self.channel_mixer = nn.Sequential(
                nn.Linear(dim, dim, bias=True),
                nn.GELU(),
                nn.Linear(dim, dim, bias=True),
            )
        elif channel_mix_type == "none":
            self.channel_mixer = nn.Identity()
        else:
            raise ValueError(f"Channel Mixer with `{channel_mix_type = }` is not supported!")

        self.sequence_norm = nn.LayerNorm(dim)
        self.channel_norm = nn.LayerNorm(dim)
        self.sequence_dropout = nn.Dropout(0.1)
        self.channel_dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Attention module.

        Parameters:
        - x: Input tensor (batch_size, seq_len_q, dim)
        - c: Context tensor (batch_size, seq_len_k, dim)

        Returns:
        - out: Output tensor after attention (batch_size, seq_len, dim)
        """

        x = self.sequence_norm(x)
        if self.sequence_mix_type == "mamba":
            x = self.sequence_mixer(x) + x
        elif self.sequence_mix_type == "attention":
            x = self.sequence_mixer(x) + x
        x = self.sequence_dropout(x)

        if self.use_cross_attention:
            # import ipdb; ipdb.set_trace()
            score = self.cross_attention(x, c)   # (batch_size, seq_len, dim)
            x = torch.cat([x, score], dim=-1)    # (batch_size, seq_len, 2 * dim)
            x = self.cross_proj(x)

        if self.channel_mix_type == "mlp":
            x = self.channel_norm(x)
            x = self.channel_mixer(x)
            x = self.channel_dropout(x)

        return x


class MambaMixer(nn.Module):
    def __init__(self, 
        dim: int, 
        len: Optional[int] = 27 * 27,
        num_blocks: int = 12, 
        sequence_mix_type: str = "mamba",
        channel_mix_type: str = "mlp"
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([
            MambaMixerBlock(
                dim,
                len,
                sequence_mix_type=sequence_mix_type,
                channel_mix_type=channel_mix_type
            ) for _ in range(num_blocks)
        ])

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, c)
        return x


class QMamba(nn.Module):
    def __init__(self,
        dim: int,
        num_queries: int = 384,
        use_mamba_encoder: bool = False
    ) -> None:
        """
        Q-Former / Querying Transformer from BLIP-2

        Args:
            dim (int): Dimensionality of the input features
            num_queries (int): Number of queries
            use_mamba_encoder (bool): Whether to use Mamba encoder

        Returns:
            torch.Tensor: Output tensor
        """
        super().__init__()
        self.queries = nn.Parameter(
            torch.zeros(1, num_queries, dim)
        )
        self.queries.data.normal_(mean=0.0, std=0.02)
        self.qmamba = MambaMixer(dim, num_blocks=2)

        self.ln_vision = nn.LayerNorm(dim)

        self.use_mamba_encoder = use_mamba_encoder
        if use_mamba_encoder:
            self.forward_mamba = Mamba(dim)
            self.backward_mamba = Mamba(dim)

    def forward(self, img_emb: torch.Tensor) -> torch.Tensor:
        img_feats = self.ln_vision(img_emb)
        if self.use_mamba_encoder:
            forward = self.forward_mamba(img_feats)
            backward = self.backward_mamba(img_feats.flip(1)).flip(1)
            img_feats = forward + backward + img_feats

        queries = self.queries.expand(img_emb.shape[0], -1, -1)
        query_output = self.qmamba(queries, img_feats)

        return query_output