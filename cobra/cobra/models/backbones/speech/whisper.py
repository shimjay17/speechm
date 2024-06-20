from dataclasses import dataclass
from functools import partial
from typing import Callable, Dict, Tuple


import torch
from torch.distributed.fsdp.wrap import _module_wrap_policy, _or_policy, transformer_auto_wrap_policy

# from workspace.samsung.cobra.cobra.models.backbones.speech.base_speech_original import HFSpeechBackbone, unpack_tuple
from cobra.models.backbones.speech.base_speech_original import HFSpeechBackbone, unpack_tuple


# Registry =>> Supported CLIP Vision Backbones (from TIMM)
WHISPER_SPEECH_BACKBONES = {
    "whisper-large": "/mnt/hdd/hsyoon/workspace/models/whisper-large-v2",
}

class WhisperBackbone(HFSpeechBackbone):
    def __init__(self, speech_backbone_id: str, speech_augmentation_strategy: str, sample_rate: int = 224) -> None:
        super().__init__(
            speech_backbone_id,
            WHISPER_SPEECH_BACKBONES[speech_backbone_id],
            speech_augmentation_strategy,
            sample_rate=sample_rate,
            override_act_layer=None # TODO: None for now
        )