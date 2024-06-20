"""
materialize.py

Factory class for initializing Vision Backbones, LLM Backbones, and VLMs from a set registry; provides and exports
individual functions for clear control flow.
"""
from typing import Optional, Tuple

from transformers import PreTrainedTokenizerBase

from cobra.models.backbones.llm import LLMBackbone, MambaLLMBackbone
from cobra.models.backbones.vision import (
    CLIPViTBackbone,
    DinoCLIPViTBackbone,
    DinoSigLIPViTBackbone,
    DinoV2ViTBackbone,
    ImageTransform,
    IN1KViTBackbone,
    SigLIPViTBackbone,
    VisionBackbone,
)
# from workspace.samsung.cobra.cobra.models.backbones.speech.base_speech_original import HFSpeechBackbone
from cobra.models.backbones.speech.base_speech_original import HFSpeechBackbone
from cobra.models.vlms import CobraVLM
from cobra.models.slms import CobraSLM

# === Registries =>> Maps ID --> {cls(), kwargs} :: Different Registries for Vision Backbones, LLM Backbones, VLMs ===
# fmt: off

# === Vision Backbone Registry ===
VISION_BACKBONES = {
    # === 224px Backbones ===
    "clip-vit-l": {"cls": CLIPViTBackbone, "kwargs": {"default_image_size": 224}},
    "siglip-vit-so400m": {"cls": SigLIPViTBackbone, "kwargs": {"default_image_size": 224}},
    "dinov2-vit-l": {"cls": DinoV2ViTBackbone, "kwargs": {"default_image_size": 224}},
    "in1k-vit-l": {"cls": IN1KViTBackbone, "kwargs": {"default_image_size": 224}},

    # === Assorted CLIP Backbones ===
    "clip-vit-b": {"cls": CLIPViTBackbone, "kwargs": {"default_image_size": 224}},
    "clip-vit-l-336px": {"cls": CLIPViTBackbone, "kwargs": {"default_image_size": 336}},

    # === Assorted SigLIP Backbones ===
    "siglip-vit-b16-224px": {"cls": SigLIPViTBackbone, "kwargs": {"default_image_size": 224}},
    "siglip-vit-b16-256px": {"cls": SigLIPViTBackbone, "kwargs": {"default_image_size": 256}},
    "siglip-vit-b16-384px": {"cls": SigLIPViTBackbone, "kwargs": {"default_image_size": 384}},
    "siglip-vit-so400m-384px": {"cls": SigLIPViTBackbone, "kwargs": {"default_image_size": 384}},

    # === Fused Backbones ===
    "dinoclip-vit-l-336px": {"cls": DinoCLIPViTBackbone, "kwargs": {"default_image_size": 336}},
    "dinosiglip-vit-so-384px": {"cls": DinoSigLIPViTBackbone, "kwargs": {"default_image_size": 384}},
}

# === speech Backbone Registry === # added by esyoon 2024-06-01-15:23:36
SPEECH_BACKBONES = {
    "whisper-large-v2": {"cls": HFSpeechBackbone, },
}

# === Language Model Registry ===
LLM_BACKBONES = {
    # === Mamba Backbones ===
    "mamba-2.8b-slimpj": {"cls": MambaLLMBackbone, "kwargs": {}},
    "mamba-2.8b": {"cls": MambaLLMBackbone, "kwargs": {}},
    "mamba-2.8b-zephyr": {"cls": MambaLLMBackbone, "kwargs": {}},
}

# fmt: on


def get_vision_backbone_and_transform(
    vision_backbone_id: str, image_resize_strategy: str
) -> Tuple[VisionBackbone, ImageTransform]:
    """Instantiate a Vision Backbone, returning both the nn.Module wrapper class and default Image Transform."""
    if vision_backbone_id in VISION_BACKBONES:
        vision_cfg = VISION_BACKBONES[vision_backbone_id]
        vision_backbone: VisionBackbone = vision_cfg["cls"](
            vision_backbone_id, image_resize_strategy, **vision_cfg["kwargs"]
        )
        image_transform = vision_backbone.get_image_transform()
        return vision_backbone, image_transform

    else:
        raise ValueError(f"Vision Backbone `{vision_backbone_id}` is not supported!")

def get_speech_backbone(
    speech_backbone_id: str, speech_backbone_path: str, speech_augmentation_strategy: str
) -> HFSpeechBackbone:
    """Instantiate a Vision Backbone, returning both the nn.Module wrapper class and default Image Transform."""
    if speech_backbone_id in SPEECH_BACKBONES:
        speech_cfg = SPEECH_BACKBONES[speech_backbone_id]
        speech_backbone: HFSpeechBackbone = speech_cfg["cls"](
            speech_backbone_id, speech_backbone_path, speech_augmentation_strategy, sample_rate=16000
        )
        processor = speech_backbone.get_speech_processor
        return speech_backbone, processor

    else:
        raise ValueError(f"speech Backbone `{speech_backbone_id}` is not supported!")


def get_llm_backbone_and_tokenizer(
    llm_backbone_id: str,
    llm_backbone_path: Optional[str] = None,
    llm_max_length: int = 2048,
    hf_token: Optional[str] = None,
    inference_mode: bool = False,
) -> Tuple[LLMBackbone, PreTrainedTokenizerBase]:
    if llm_backbone_id in LLM_BACKBONES:
        llm_cfg = LLM_BACKBONES[llm_backbone_id]
        llm_backbone: LLMBackbone = llm_cfg["cls"](
            llm_backbone_id,
            llm_max_length=llm_max_length,
            hf_token=hf_token,
            inference_mode=inference_mode,
            llm_backbone_path=llm_backbone_path,
            **llm_cfg["kwargs"],
        )
        tokenizer = llm_backbone.get_tokenizer()
        pre_additional_special_tokens = tokenizer.additional_special_tokens
        pre_additional_special_tokens = pre_additional_special_tokens + ["<Speech>", "</Speech>"]
        tokenizer.add_special_tokens(
            {"additional_special_tokens":pre_additional_special_tokens}
        )
        llm_backbone.llm.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)
        return llm_backbone, tokenizer

    else:
        raise ValueError(f"LLM Backbone `{llm_backbone_id}` is not supported!")


def get_vlm(
    model_id: str,
    arch_specifier: str,
    vision_backbone: VisionBackbone,
    # speech_backbone: SpeechBackbone,
    llm_backbone: LLMBackbone,
    enable_mixed_precision_training: bool = True,
):
    """Lightweight wrapper around initializing a VLM, mostly for future-proofing (if one wants to add a new VLM)."""
    return CobraVLM(
        model_id,
        vision_backbone,
        llm_backbone,
        enable_mixed_precision_training=enable_mixed_precision_training,
        arch_specifier=arch_specifier,
    )


def get_slm(
    model_id: str,
    arch_specifier: str,
    speech_backbone: HFSpeechBackbone,
    llm_backbone: LLMBackbone,
    second_per_window: float,
    second_stride: float,
    enable_mixed_precision_training: bool = True,
):
    """Lightweight wrapper around initializing a VLM, mostly for future-proofing (if one wants to add a new VLM)."""
    return CobraSLM(
        model_id,
        speech_backbone,
        llm_backbone,
        second_per_window=second_per_window,
        second_stride=second_stride,
        enable_mixed_precision_training=enable_mixed_precision_training,
        arch_specifier=arch_specifier,
    )
