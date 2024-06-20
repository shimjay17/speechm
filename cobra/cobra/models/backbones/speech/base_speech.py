from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, Optional, Protocol, Tuple, Union

import torch
import torch.nn as nn

from torch.distributed.fsdp.wrap import _module_wrap_policy, _or_policy, transformer_auto_wrap_policy
from transformers import (
    WhisperFeatureExtractor,
    WhisperModel,
)

# === Utility Functions for Monkey-Patching ===
def unpack_tuple(fn: Callable[[Any], Tuple[Any]]) -> Callable[[Any], Any]:
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        result = fn(*args, **kwargs)
        return result[0] if isinstance(result, tuple) else result

    return wrapper


# === Abstract Base Class for arbitrary Vision Backbones ===
class SpeechBackbone(nn.Module, ABC):
    def __init__(self, speech_backbone_id: str, speech_augmentation_strategy: str, sample_rate: int = 16_000) -> None:
        super().__init__()
        self.identifier: str = speech_backbone_id
        self.speech_augmentation_strategy: str = speech_augmentation_strategy
        self.sample_rate: int = sample_rate

        # Instance attributes for a Vision Backbone
        self.featurizer: nn.Module = None
        # self.image_transform: ImageTransform = None

    # def get_image_transform(self) -> ImageTransform:
    #     return self.image_transform

    @abstractmethod
    def get_fsdp_wrapping_policy(self) -> Callable: ...

    @abstractmethod 
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor: # TODO: check the name input_ids fine
        """Run a forward pass through the featurizer given a set of processed images, returning patch/grid features."""
        raise NotImplementedError

    @property
    @abstractmethod
    def default_image_resolution(self) -> Tuple[int, int, int]: ...

    @property
    @abstractmethod
    def embed_dim(self) -> int: ...

    @property
    @abstractmethod
    def num_patches(self) -> int: ...

    @property
    @abstractmethod
    def half_precision_dtype(self) -> torch.dtype: ...


class HFSpeechBackbone(SpeechBackbone, ABC):
    def __init__(
        self,
        speech_backbone_id: str,
        model_name_or_path: str,
        speech_augmentation_strategy: str,
        sample_rate: int = 16000,
        override_act_layer: Optional[str] = None,
    ) -> None:
        super().__init__(speech_backbone_id, speech_augmentation_strategy, sample_rate=sample_rate)
        self.model_name_or_path = model_name_or_path
        self.override_act_layer = override_act_layer
        self.dtype = torch.bfloat16

        # Initialize Featurizer (ViT) by downloading from HF / TIMM Hub if necessary
        if self.override_act_layer is None:
            self.featurizer = WhisperFeatureExtractor.from_pretrained(self.model_name_or_path)

            # whisper
            self.speech_encoder = WhisperModel.from_pretrained(self.model_name_or_path).encoder
            self.ln_speech = nn.LayerNorm(self.speech_encoder.config.d_model) # TODO: It does not need because there are layernorm after speech encoder

        # if self.override_act_layer is None:
        #     import ipdb; ipdb.set_trace()
        #     self.featurizer: VisionTransformer = timm.create_model(
        #         self.timm_path_or_url, pretrained=True, num_classes=0, img_size=self.default_image_size,
        #     )
        # else:
        #     self.featurizer: VisionTransformer = timm.create_model(
        #         self.timm_path_or_url,
        #         pretrained=True,
        #         num_classes=0,
        #         img_size=self.default_image_size,
        #         act_layer=self.override_act_layer,
        #     )
        #self.featurizer.eval()
        self.featurizer.eval()

        # # Monkey-Patch the `forward()` function of the featurizer to ensure FSDP-compatibility
        # #   => Note: By default set `get_intermediate_layers` to return the *SECOND-TO-LAST* layer patches!
        # #   => TODO (siddk) Remove after resolution of https://github.com/pytorch/pytorch/issues/109385
        # TODO: what is moneky packing in here?
        self.featurizer.forward = unpack_tuple(
            partial(self.featurizer.get_intermediate_layers, n={len(self.featurizer.blocks) - 2})
        )

        # # Validation =>> for now, this class *only* supports TIMM Vision Transformers (but can be extended!)
        # # assert isinstance(self.featurizer, VisionTransformer), (
        # #     "Featurizer is not a TIMM VisionTransformer; if you would like to support a new visual representation, "
        # #     "file an issue or implement the requisite logic (see `cobra/models/backbones/vision/base_vision.py`)!"
        # # )

        # # Get Config =>> Note :: Override default image size to ensure correct image transform
        # self.data_cfg = timm.data.resolve_model_data_config(self.featurizer)
        # self.data_cfg["input_size"] = (3, self.default_image_size, self.default_image_size)

        # # Initialize Default Image Transform --> Modified by `self.image_resize_strategy`
        # default_image_transform = timm.data.create_transform(**self.data_cfg, is_training=False)

        # # Fix =>> SigLIP & IN1K default transforms resize to *larger* than `self.default_image_size` (crops image)!
        # if "siglip" in self.timm_path_or_url or "in1k" in self.timm_path_or_url:
        #     assert isinstance(default_image_transform, Compose), "Unexpected `default_image_transform`!"
        #     assert isinstance(resize_transform := default_image_transform.transforms[0], Resize)
        #     default_image_transform = Compose(
        #         [
        #             Resize(self.default_image_size, interpolation=resize_transform.interpolation),
        #             *default_image_transform.transforms[1:],
        #         ]
        #     )

        # # Switch on `image_resize_strategy`
        # if self.image_resize_strategy == "resize-naive":
        #     assert isinstance(default_image_transform, Compose), "Unexpected `default_image_transform`!"
        #     assert isinstance(resize_transform := default_image_transform.transforms[0], Resize)

        #     target_size = (self.default_image_size, self.default_image_size)
        #     self.image_transform = Compose(
        #         [
        #             Resize(target_size, interpolation=resize_transform.interpolation),
        #             *default_image_transform.transforms[1:],
        #         ]
        #     )

        # elif self.image_resize_strategy == "resize-crop":
        #     self.image_transform = default_image_transform

        # elif self.image_resize_strategy == "letterbox":
        #     assert isinstance(default_image_transform, Compose), "Unexpected `default_image_transform`!"
        #     assert "mean" in self.data_cfg, "TIMM `data_cfg` missing image normalization mean!"

        #     # Compute Padding Fill Value (rescaled normalization mean if applicable)
        #     fill = tuple([int(x * 255) for x in self.data_cfg["mean"]])

        #     # Build New Transform
        #     self.image_transform = Compose([LetterboxPad(fill), *default_image_transform.transforms])

        # else:
        #     raise ValueError(f"Image Resize Strategy `{self.image_resize_strategy}` is not supported!")

    def get_fsdp_wrapping_policy(self) -> Callable:
        """Return a simple FSDP policy that wraps each ViT block and then the _entire_ featurizer."""
        vit_wrap_policy = partial(_module_wrap_policy, module_classes={VisionTransformer})
        transformer_block_policy = partial(transformer_auto_wrap_policy, transformer_layer_cls={Block})
        return partial(_or_policy, policies=[vit_wrap_policy, transformer_block_policy])

    def forward(self, pixel_values: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> torch.Tensor:
        """Runs transformed image/pixel tensor through vision backbone, returning _all_ patch features."""
        return self.featurizer(pixel_values)

    @property
    def default_image_resolution(self) -> Tuple[int, int, int]:
        return self.data_cfg["input_size"]

    @property
    def embed_dim(self) -> int:
        return self.featurizer.embed_dim

    @property
    def num_patches(self) -> int:
        return self.featurizer.patch_embed.num_patches

    @property
    def half_precision_dtype(self) -> torch.dtype:
        return self.dtype
