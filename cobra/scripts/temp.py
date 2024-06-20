import requests
import torch

from PIL import Image
from pathlib import Path

from cobra import load

hf_token = Path(".hf_token").read_text().strip()
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# In case your GPU does not support bf16
dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

# Load a pretrained VLM (either local path, or ID to auto-download from the HF Hub) 
model_path = "/data2/mamba/workspace/neurips24/huggingface/cobra/cobra+3b"
vlm = load(model_path, hf_token=hf_token, llm_only=True, arch_specifier="qmamba")
vlm.to(device, dtype=dtype)

print(vlm.vision_backbone.embed_dim)
print(vlm.llm_backbone.embed_dim)

total_params = sum(p.numel() for p in vlm.parameters())
train_params = sum(p.numel() for p in vlm.parameters() if p.requires_grad)
proj_params = sum(p.numel() for p in vlm.projector.parameters())
print(f"Total Parameters = `{total_params / 1e9 :.2f}B`")
print(f"Trainable Parameters = `{train_params / 1e9 :.2f}B`")
print(f"Projector Parameters = `{proj_params / 1e6:.2f}M`")
