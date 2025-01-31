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
model_path = "/data/soohwan/workspace/huggingface/cobra/cobra+3b"
vlm = load(model_path, hf_token=hf_token, llm_only=True, arch_specifier="mamba")
vlm.to(device, dtype=dtype)

# Download an image and specify a prompt
image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png"
image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")
user_prompt = "What is happening in this image?"

# Build prompt
prompt_builder = vlm.get_prompt_builder()
prompt_builder.add_turn(role="human", message=user_prompt)
prompt_text = prompt_builder.get_prompt()

# Generate!
generated_text = vlm.generate(
    image,
    prompt_text,
    cg=True,
    do_sample=True,
    temperature=0.4,
    max_new_tokens=512,
)

print(generated_text)