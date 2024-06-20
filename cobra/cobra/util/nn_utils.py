"""
nn_utils.py

Utility functions and PyTorch submodule definitions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math
from cobra.models.mamba.modeling_mamba import Mamba, MambaForCausalLM
# from cobra.utils.qmamba import QMamba
from functools import partial

from flash_attn import flash_attn_func as flash_attn
from transformers import AutoTokenizer

# === Definitions for Various Projection Modules, with Signature :: [..., in_dim] --> [..., out_dim] ===
class LinearProjector(nn.Module):
    def __init__(self, vision_dim: int, llm_dim: int) -> None:
        super().__init__()
        self.projector = nn.Linear(vision_dim, llm_dim, bias=True)

    def forward(self, img_patches: torch.Tensor) -> torch.Tensor:
        return self.projector(img_patches)


class MLPProjector(nn.Module):
    def __init__(self, vision_dim: int, llm_dim: int, mlp_type: str = "gelu-mlp") -> None:
        super().__init__()
        if mlp_type == "gelu-mlp":
            self.projector = nn.Sequential(
                nn.Linear(vision_dim, llm_dim, bias=True),
                nn.GELU(),
                nn.Linear(llm_dim, llm_dim, bias=True),
            )
        else:
            raise ValueError(f"Projector with `{mlp_type = }` is not supported!")

    def forward(self, img_patches: torch.Tensor) -> torch.Tensor:
        return self.projector(img_patches)


class FusedMLPProjector(nn.Module):
    def __init__(self, fused_vision_dim: int, llm_dim: int, mlp_type: str = "fused-gelu-mlp") -> None:
        super().__init__()
        self.initial_projection_dim = fused_vision_dim * 4
        if mlp_type == "fused-gelu-mlp":
            self.projector = nn.Sequential(
                nn.Linear(fused_vision_dim, self.initial_projection_dim, bias=True),
                nn.GELU(),
                nn.Linear(self.initial_projection_dim, llm_dim, bias=True),
                nn.GELU(),
                nn.Linear(llm_dim, llm_dim, bias=True),
            )
        else:
            raise ValueError(f"Fused Projector with `{mlp_type = }` is not supported!")

    def forward(self, fused_img_patches: torch.Tensor) -> torch.Tensor:
        return self.projector(fused_img_patches)


# LDPv2 Projector: https://github.com/Meituan-AutoML/MobileVLM/blob/main/mobilevlm/model/vision_projector.py
class TokenDownLayer(nn.Module):
    def __init__(self, shape) -> None:
        super().__init__()
        self.dwn = nn.Sequential(
            nn.AdaptiveAvgPool2d(shape)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, num_tokens, c = x.shape
        h = int(math.sqrt(num_tokens))
        assert h * h == num_tokens
        x = x.permute(0, 2, 1).reshape(b, -1, h, h)
        x = self.dwn(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class PosInjectLayer(nn.Module):
    # https://github.com/Meituan-AutoML/Twins/blob/main/gvt.py
    def __init__(self, in_dim: int, out_dim: int, stride: int = 1) -> None:
        super().__init__()
        self.peg = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 3, stride, 1, bias=True, groups=out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, num_tokens, c = x.shape
        h = int(math.sqrt(num_tokens))
        assert h * h == num_tokens
        cnn_feat = x.transpose(1, 2).view(b, c, h, h)
        x = self.peg(cnn_feat) + cnn_feat
        x = x.flatten(2).transpose(1, 2)
        return x
       

class LDPProjector(nn.Module):
    def __init__(self, vision_dim: int, llm_dim: int, mlp_type: str = "ldpnet") -> None:
        super().__init__()
        if mlp_type == "ldpnet":
            self.projector = nn.Sequential(
                nn.Linear(vision_dim, llm_dim, bias=True),
                nn.GELU(),
                nn.Linear(llm_dim, llm_dim, bias=True),
                TokenDownLayer((14, 14)),
                PosInjectLayer(llm_dim, llm_dim, stride=1),
            )
        else:
            raise ValueError(f"Projector with `{mlp_type = }` is not supported!")

    def forward(self, img_patches: torch.Tensor) -> torch.Tensor:
        return self.projector(img_patches)
    
        
class FusedLDPProjector(nn.Module):
    def __init__(self, fused_vision_dim: int, llm_dim: int, mlp_type: str = "fused-ldpnet") -> None:
        super().__init__()
        self.initial_projection_dim = fused_vision_dim * 4
        if mlp_type == "fused-ldpnet":
            self.projector = nn.Sequential(
                nn.Linear(fused_vision_dim, self.initial_projection_dim, bias=True), 
                nn.GELU(), 
                nn.Linear(self.initial_projection_dim, llm_dim, bias=True),
                TokenDownLayer((14, 14)),
                PosInjectLayer(llm_dim, llm_dim, stride=1),
            )
            
        else:
            raise ValueError(f"Fused Projector with `{mlp_type = }` is not supported!")
        
    def forward(self, img_patches: torch.Tensor) -> torch.Tensor:
        return self.projector(img_patches)

#################### added by sheom 2024.04.27 ####################

class ResidualBidirectionalMambaLayer(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.ForwardMamba = Mamba(dim)
        self.BackwardMamba = Mamba(dim)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # import ipdb; ipdb.set_trace()
        forward = self.ForwardMamba(x)
        backward = self.BackwardMamba(x.flip(1)).flip(1)
        return forward + backward + x


class MambaProjectorV2(nn.Module):
    """
    Multimodal Connector from VL-Mamba (https://arxiv.org/abs/2403.13600).
    """
    def __init__(self, fused_vision_dim: int, llm_dim: int, mlp_type: str = "mamba") -> None:
        super().__init__()
        if mlp_type == "mamba":
            self.projector = nn.Sequential(
                nn.Linear(fused_vision_dim, llm_dim, bias=True),
                nn.GELU(),
                ResidualBidirectionalMambaLayer(llm_dim),
                nn.LayerNorm(llm_dim),
                nn.Linear(llm_dim, llm_dim, bias=True),
            )
        else:
            raise ValueError(f"Projector with `{mlp_type = }` is not supported!")

    def forward(self, img_patches: torch.Tensor) -> torch.Tensor:
        return self.projector(img_patches)
    
class MambaProjectorV3(nn.Module):
    """
    Multimodal Connector from VL-Mamba (https://arxiv.org/abs/2403.13600).
    """
    def __init__(self, fused_vision_dim: int, llm_dim: int, mlp_type: str = "mamba", length: int=1500) -> None:
        super().__init__()
        if mlp_type == "mamba":
            self.projector = nn.Sequential(
                nn.Linear(fused_vision_dim, llm_dim, bias=True),
                nn.GELU(),
                ResidualBidirectionalMambaLayer(llm_dim),
                nn.LayerNorm(llm_dim),
                nn.Linear(llm_dim, llm_dim, bias=True),
                nn.GELU(),
                nn.Conv1d(length, 500, kernel_size=1, stride=1, padding=0),
            )
        else:
            raise ValueError(f"Projector with `{mlp_type = }` is not supported!")

    def forward(self, img_patches: torch.Tensor) -> torch.Tensor:
        return self.projector(img_patches)

################### added by sheom 2024.05.03 ####################

class Attention(nn.Module):
    def __init__(self,
        dim: int,
        num_heads: int = 16,
        bias: bool = True,
        dropout: float = 0.0,
        batch_first: bool = True,
        use_flash: bool = False,
        local: bool = True,
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
        self.local = local

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

            if self.local:
                seqlen_q = x.size(1)
                seqlen_k = c.size(1)
                window_size = (seqlen_k - seqlen_q, 0)
            else:
                window_size = (-1, -1)

            # Apply flash attention
            out = flash_attn(Q, K, V,
                dropout_p=self.dropout_p if self.training else 0.0,
                causal=False,
                window_size=window_size
            )   # (batch_size, seq_len_q, num_heads, head_dim)
            
            # Transpose and reshape back to (batch_size, seq_len_q, dim)
            out = out.contiguous().view(batch_size, seq_len, dim)

            # Final linear projection
            out = self.proj(out)
            out = self.proj_drop(out)

            return out
        else:
            # Multihead attention requires (seq_len, batch_size, dim)
            if self.local:
                seqlen_q = x.size(1)
                seqlen_k = c.size(1)
                mask = self.get_local_mask(seqlen_q, seqlen_k).to(x.device)
            else:
                mask = None
            out, _ = self.attn(x, c, c, attn_mask = mask)

            return out
    
    def get_local_mask(self, num_q: int, num_k: int) -> torch.Tensor:
        # Calculate how many keys each query normally attends to
        keys_per_query = num_k // num_q
        extra_keys = num_k % num_q

        # Initialize the attention mask with zeros
        attention_mask = torch.ones(num_q, num_k)

        # Fill in the attention mask such that each query attends to a unique set of keys
        start_key = 0
        for i in range(num_q):
            # Adjust the range to include an extra key for the first 'extra_keys' queries
            end_key = start_key + keys_per_query + (1 if i < extra_keys else 0)
            attention_mask[i, start_key:end_key] = 0
            start_key = end_key

        return attention_mask.bool()


class MambaMixerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        sequence_mix_type: str = "mamba",
        channel_mix_type: str = "mlp", 
        use_cross_attention: bool = True,
        dropout_p: float = 0.0
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

        # for text input
        self.text_ffn = nn.Sequential(
            nn.Linear(dim, dim, bias=True),
            nn.GELU(),
            nn.Linear(dim, dim, bias=True)
        )
        self.text_norm = nn.LayerNorm(dim)
        self.text_dropout = nn.Dropout(dropout_p)

        self.sequence_norm = nn.LayerNorm(dim)
        self.channel_norm = nn.LayerNorm(dim)
        self.sequence_dropout = nn.Dropout(dropout_p)
        self.channel_dropout = nn.Dropout(dropout_p)

    def forward(self, x: torch.Tensor, c: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass of the Attention module.

        Parameters:
        - x: Input tensor (batch_size, seq_len_q, dim)
        - c: Context tensor (batch_size, seq_len_k, dim)

        Returns:
        - out: Output tensor after attention (batch_size, seq_len, dim)
        """

        assert c is not None or not self.use_cross_attention, "Context tensor required for cross-attention!"

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
            x = self.channel_mixer(x) + x
            x = self.channel_dropout(x)

        return x

    def forward_text(self, x: torch.Tensor) -> torch.Tensor:
        x = self.sequence_norm(x)
        if self.sequence_mix_type == "mamba":
            x = self.sequence_mixer(x) + x
        elif self.sequence_mix_type == "attention":
            x = self.sequence_mixer(x) + x
        x = self.sequence_dropout(x)

        x = self.text_norm(x)
        x = self.text_ffn(x)
        x = self.text_dropout(x)
        
        return x


class MambaMixer(nn.Module):
    def __init__(self, 
        dim: int, 
        num_blocks: int = 12, 
        sequence_mix_type: str = "mamba",
        channel_mix_type: str = "mlp",
        use_cross_attention: bool = True
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([
            MambaMixerBlock(
                dim,
                sequence_mix_type=sequence_mix_type,
                channel_mix_type=channel_mix_type,
                use_cross_attention=use_cross_attention
            ) for _ in range(num_blocks)
        ])
    def forward(self, x: torch.Tensor, c: torch.Tensor = None) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, c)
        return x

    def forward_text(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block.forward_text(x)
        return x


class QMamba(nn.Module):
    def __init__(self,
        input_dim: int,
        dim: int,
        num_queries: int = 384,
        num_blocks: int = 24,
        sequence_mix_type: str = "mamba",
        channel_mix_type: str = "none",
        use_mamba_encoder: bool = True,
        load_from_pretrained: bool = False,
        image_text_pretraining: bool = False
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
        self.dim = dim
        self.num_blocks = num_blocks
        self.queries = nn.Parameter(
            torch.zeros(1, num_queries, dim)
        )
        self.queries.data.normal_(mean=0.0, std=0.02)
        self.qmamba = MambaMixer(
            dim,
            num_blocks=num_blocks,
            sequence_mix_type=sequence_mix_type,
            channel_mix_type=channel_mix_type,
            use_cross_attention=True
        )

        self.ln_vision = nn.LayerNorm(dim)

        self.use_mamba_encoder = use_mamba_encoder
        if use_mamba_encoder:
            self.forward_mamba = Mamba(dim)
            self.backward_mamba = Mamba(dim)

        if load_from_pretrained:
            self.load_from_pretrained_mamba(self.pretrained)

        self.vision_proj = nn.Linear(input_dim, dim, bias=True)

        self.pretrained = "state-space/mamba-130m-hf"

        if image_text_pretraining:
            self.embedding = nn.Embedding(50280, dim)
            self.temperature = 0.07

    def forward(self, img_emb: torch.Tensor) -> torch.Tensor:
        img_feats = self.vision_proj(img_emb)
        img_feats = self.ln_vision(img_feats)
        if self.use_mamba_encoder:
            forward = self.forward_mamba(img_feats)
            backward = self.backward_mamba(img_feats.flip(1)).flip(1)
            img_feats = forward + backward + img_feats

        queries = self.queries.expand(img_emb.shape[0], -1, -1)
        query_output = self.qmamba(queries, img_feats)

        return query_output

    def load_from_pretrained_mamba(self, pretrained) -> None:
        pretrained_mamba = MambaForCausalLM.from_pretrained(pretrained)
        pretrained_config = pretrained_mamba.config
        assert pretrained_config.d_model == self.dim, f"Dimension mismatch! {pretrained_config.d_model} != {self.dim}"
        assert pretrained_config.num_layers == self.num_blocks, f"Number of layers mismatch! {pretrained_config.num_layers} != {self.num_blocks}"

        # Load the weights
        # copy Mamba weights from each layer to the corresponding QMamba layer
        print(f"Loading weights from {pretrained} to Q-Mamba...")
        for i, block in enumerate(self.qmamba.blocks):
            pretrained_block = pretrained_mamba.backbone.layers[i]
            block.sequence_mixer.load_state_dict(pretrained_block.mixer.state_dict())

        print("Weights loaded successfully!")

    def forward_text(self, text: torch.Tensor) -> torch.Tensor:
        assert self.tokenizer is not None, "Tokenizer not initialized!"

        text = self.tokenizer(text, return_tensors="pt")
        text_emb = self.embedding(text["input_ids"])
        text_output = self.qmamba.forward_text(text_emb)

        # length = position of <eos> token
        # note that padding idx is also <eos>
        text_lengths = text["lengths"]

        return text_output, text_lengths

    def forward_pretrain(self, img_emb: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        query_output = self.forward(img_emb)
        img_feats = F.normalize(query_output, dim=-1)
        text_output, text_lengths = self.forward_text(text)
        txt_feats = F.normalize(text_output, dim=-1)

        # Image-Text Contrastive Pre-Training
        # We want to compute the cosine similarity between the image and text features
        # Since Mamba outputs are causal 1D sequence, we compute between the last tokens
        # Note that text lengths are not fixed, so we need to handle this case

        img_last_feats = img_feats[:, -1, :]                                            # (batch_size, dim)
        txt_last_feats = txt_feats[torch.arange(txt_feats.size(0)), text_lengths, :]    # (batch_size, dim)

        # since the features are normalized, we can directly compute cosine similarity with dot product
        # similarity matirx = batch_size x batch_size matrix
        similarity = torch.matmul(img_last_feats, txt_last_feats.T)                    # (batch_size, batch_size)
        logits = similarity / self.temperature
        
        # create labels for contrastive loss
        labels = torch.arange(similarity.size(0)).to(similarity.device)

        # compute contrastive loss
        loss_i = F.cross_entropy(similarity, labels)
        loss_t = F.cross_entropy(similarity.T, labels)

        return (loss_i + loss_t) / 2.


class QMambaProjector(nn.Module):
    def __init__(self, fused_vision_dim: int, llm_dim: int, mlp_type: str = "qmamba", inner_dim: int = 768) -> None:
        super().__init__()
        if mlp_type == "qmamba":
            self.projector = nn.Sequential(
                QMamba(fused_vision_dim, inner_dim),
                nn.Linear(inner_dim, llm_dim, bias=True),
            )
        else:
            raise ValueError(f"Projector with `{mlp_type = }` is not supported!")
    def forward(self, img_patches: torch.Tensor) -> torch.Tensor:
        return self.projector(img_patches)

#################### added by sheom 2024.05.21 ####################

class SimpleQMamba(nn.Module):
    def __init__(self, dim: int, vision_dim: int = 2560, query_length: int = 512) -> None:
        super().__init__()

        # Querying-Mamba
        self.queries = nn.Parameter(
            torch.zeros(1, query_length, dim)
        )
        self.sequence_mixer = Mamba(dim)

        # Cross-Attention
        self.cross_attention = Attention(dim)

        # FFN Layer
        self.channel_mixer = nn.Sequential(
            nn.Linear(dim, dim, bias=True),
            nn.GELU(),
            nn.Linear(dim, dim, bias=True),
        )
        self.sequence_norm = nn.LayerNorm(dim)
        self.channel_norm = nn.LayerNorm(dim)
        
    def forward(self, img_feats: torch.Tensor) -> torch.Tensor:

        # Querying-Mamba
        queries = self.queries.expand(img_feats.shape[0], -1, -1)
        queries = self.sequence_norm(queries)
        queries = self.sequence_mixer(queries) + queries

        # Cross-Attention
        score = self.cross_attention(queries, img_feats)
        z = score + queries

        # FFN Layer
        z = self.channel_norm(z)
        z = self.channel_mixer(z) + z

        return z


class MultiHeadSimpleQMamba(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, query_length: int = 512) -> None:
        super().__init__()

        self.qmambas = nn.ModuleList([
            SimpleQMamba(dim, query_length=query_length) for _ in range(num_heads)
        ])

    def forward(self, img_feats: torch.Tensor) -> torch.Tensor:
        head_outputs = [qmamba(img_feats) for qmamba in self.qmambas]
        return torch.cat(head_outputs, dim=-1)


class SimpleQMambaProjector(nn.Module):
    def __init__(
        self,
        fused_vision_dim: int,
        llm_dim: int,
        inner_dim: int = 2560,
        num_heads: int = 2,
        mlp_type: str = "qmamba",
        query_length: int = 256
    ) -> None:
        super().__init__()
        if mlp_type == "qmamba":
            self.projector = MambaProjectorV2(fused_vision_dim, llm_dim)
            self.vision_norm = nn.LayerNorm(llm_dim)
            self.vision_proj = nn.Linear(llm_dim, inner_dim, bias=True)

            self.qmamba = SimpleQMamba(llm_dim, query_length=query_length)
            # self.qmamba = MultiHeadSimpleQMamba(inner_dim, num_heads=num_heads, query_length=256)
            if inner_dim * num_heads != llm_dim:
                self.head_proj = nn.Linear(inner_dim * num_heads, llm_dim, bias=True)
            else:
                self.head_proj = nn.Identity()

            # self.load_from_vlmamba()
            # make sure the projector is not frozen
            for param in self.projector.parameters():
                param.requires_grad = True
        else:
            raise ValueError(f"Projector with `{mlp_type = }` is not supported!")
    def forward(self, img_patches: torch.Tensor) -> torch.Tensor:
        img_feats = self.projector(img_patches)
        img_feats = self.vision_norm(img_feats)
        img_feats = self.vision_proj(img_feats)

        proj_feats = self.qmamba(img_feats)
        # proj_feats = self.head_proj(proj_feats)
        return proj_feats

    def load_from_vlmamba(self) -> None:
        pretrained_projector = torch.load("/data2/mamba/workspace/neurips24/cobra/runs/vlmamba-finetune+siglip+ln/checkpoints/latest-checkpoint.pt")["model"]["projector"]
        self.projector.load_state_dict(pretrained_projector)
        # freeze the projector
        for param in self.projector.parameters():
            param.requires_grad = False

    