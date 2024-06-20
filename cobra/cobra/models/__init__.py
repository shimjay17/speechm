

from pathlib import Path
from typing import Optional

# from vlm_eval.util.interfaces import VLM

# from .instructblip import InstructBLIP
# from .llava import LLaVa
# from .prismatic import PrismaticVLM
from .slms import CobraSLM, SLM


# from .cobra import CobraVLM
# from .. import available_model_names, available_models, get_model_description, load

from .load_speech import available_model_names, available_models, get_model_description, load
from .materialize_speech import get_llm_backbone_and_tokenizer, get_speech_backbone, get_slm # added by esyoon 2024-06-01-17:20:10

# === Initializer Dispatch by Family ===
# FAMILY2INITIALIZER = {"instruct-blip": InstructBLIP, "llava-v15": LLaVa, "prismatic": PrismaticVLM, "mamba": mambaVLM}
# FAMILY2INITIALIZER = {"instruct-blip": InstructBLIP, "llava-v15": LLaVa, "mamba": mambaVLM, "cobra": CobraVLM}
# FAMILY2INITIALIZER = {"instruct-blip": InstructBLIP, "llava-v15": LLaVa, "mamba": mambaVLM}

#added by Jay Shim 2024-06-01-17:20:10
# =============================================================================================================
from .my_cobra_speech_eval import CobraSLMEval, SLMEval
FAMILY2INITIALIZER = {'mamba': CobraSLMEval} #added by Jay Shim 2024-06-01-17:20:10

def load_slm(
    model_family: str,
    model_id: str,
    run_dir: Path,
    hf_token: Optional[str] = None,
    ocr: Optional[bool] = False,
    load_precision: str = "bf16",
    max_length=128,
    temperature=0.4,
) -> SLMEval:
    assert model_family in FAMILY2INITIALIZER, f"Model family `{model_family}` not supported!"
    return FAMILY2INITIALIZER[model_family](
        model_family=model_family,
        model_id=model_id,
        run_dir=run_dir,
        hf_token=hf_token,
        load_precision=load_precision,
        max_length=max_length,
        temperature=temperature,
        ocr=ocr,
    )


        # self,
        # model_id: str,
        # speech_backbone: HFSpeechBackbone,
        # llm_backbone: MambaLLMBackbone,
# =============================================================================================================