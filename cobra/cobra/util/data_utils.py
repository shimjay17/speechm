"""
data_utils.py

General utilities and classes for facilitating data loading and collation.
"""
from dataclasses import dataclass
from typing import Dict, Sequence, Tuple, Union, List, Optional

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import WhisperProcessor, PreTrainedTokenizerBase
import numpy as np

def exact_div(x, y):
    assert x % y == 0
    return x // y

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100




@dataclass
class PaddedCollatorForLanguageModeling:
    model_max_length: int
    pad_token_id: int
    default_image_resolution: Tuple[int, int, int]
    padding_side: str = "right"
    pixel_values_dtype: torch.dtype = torch.float32

    def __post_init__(self) -> None:
        self.dummy_pixel_values = torch.zeros(self.default_image_resolution, dtype=self.pixel_values_dtype)

    def __call__(self, instances: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        pixel_values = [instance["pixel_values"] for instance in instances]

        # For now, we only support Tokenizers with `padding_side = "right"` during Training (but plan to extend!)
        #   => Handle padding via RNN Utils => `pad_sequence`
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)

        # Truncate (if necessary)
        input_ids, labels = input_ids[:, : self.model_max_length], labels[:, : self.model_max_length]

        # Get `attention_mask` by checking for `pad_token_id`
        attention_mask = input_ids.ne(self.pad_token_id)

        # === Handle "unimodal" (language-only) vs. "multimodal" ===

        # Some examples are "language-only" --> build a Tensor of `multimodal_indices` that we can slice into easily
        multimodal_indices = torch.tensor(
            [idx for idx in range(len(pixel_values)) if pixel_values[idx] is not None], dtype=torch.long
        )

        # Stack all `pixel_values` --> depending on type (torch.Tensor, or Dict[str, torch.Tensor]) & presence of None
        if len(multimodal_indices) == 0:
            pixel_values = torch.stack([self.dummy_pixel_values for _ in range(len(input_ids))])
        elif isinstance(pv_example := pixel_values[multimodal_indices[0]], torch.Tensor):
            pixel_values = torch.stack(
                [
                    pixel_values[idx] if idx in multimodal_indices else self.dummy_pixel_values
                    for idx in range(len(input_ids))
                ]
            )
        elif isinstance(pv_example, dict):
            pixel_values = {
                k: torch.stack(
                    [
                        pixel_values[idx][k] if idx in multimodal_indices else self.dummy_pixel_values
                        for idx in range(len(input_ids))
                    ]
                )
                for k in pv_example
            }
        else:
            raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")

        return dict(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            multimodal_indices=multimodal_indices,
        )

# hard coded for now for audio
SAMPLE_RATE = 16000
N_FFT = 400
N_MELS = 80
HOP_LENGTH = 160
CHUNK_LENGTH = 30
N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE  # 480000: number of samples in a chunk
N_FRAMES = exact_div(N_SAMPLES, HOP_LENGTH)  # 3000: number of frames in a mel spectrogram input


@dataclass
class SpeechPaddedCollatorForLanguageModeling:
    processor: WhisperProcessor
    tokenizer: PreTrainedTokenizerBase
    model_max_length: int
    padding: Union[bool, str]
    pad_token_id: int
    sample_rate: int = 16000
    padding_side: str = "right"
    input_values_dtype: torch.dtype = torch.float32

    def __post_init__(self) -> None:
        # TODO: 확인 필요
        self.dummy_input_values = torch.zeros(self.model_max_length, dtype=self.input_values_dtype)
        self.speech_start_token = self.tokenizer("<Speech>", add_special_tokens=False)["input_ids"][0]
        self.speech_end_token = self.tokenizer("</Speech>", add_special_tokens=False)["input_ids"][0]

    def __call__(self, instances: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_values = [instance["input_values"] for instance in instances]
        assert self.sample_rate == instances[0]['sample_rate']

        input_features = self.processor(
            audio=input_values,
            sampling_rate=self.sample_rate,
            padding=self.padding,
            # max_length=self.model_max_length,
            return_tenors="pt").input_features
        
        input_features = np.stack(input_features, axis=0)
        input_features = torch.tensor(input_features, dtype=torch.float32)
    

        # For now, we only support Tokenizers with `padding_side = "right"` during Training (but plan to extend!)
        #   => Handle padding via RNN Utils => `pad_sequence`
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)

        # Truncate (if necessary)
        input_ids, labels = input_ids[:, : self.model_max_length], labels[:, : self.model_max_length]

        # Get `attention_mask` by checking for `pad_token_id`
        attention_mask = input_ids.ne(self.pad_token_id)
        # === Handle "unimodal" (language-only) vs. "multimodal" ===

        # Some examples are "language-only" --> build a Tensor of `multimodal_indices` that we can slice into easily
        multimodal_indices = torch.tensor(
            [idx for idx in range(len(input_values)) if input_values[idx] is not None], dtype=torch.long
        )
        return dict(
            input_values=input_features,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            multimodal_indices=multimodal_indices,
        )

@dataclass
class GigaSpeechPaddedCollatorForLanguageModeling:
    processor: WhisperProcessor
    tokenizer: PreTrainedTokenizerBase
    model_max_length: int
    padding: Union[bool, str]
    pad_token_id: int
    sample_rate: int = 16000
    padding_side: str = "right"
    input_values_dtype: torch.dtype = torch.float32

    def __post_init__(self) -> None:
        # TODO: 확인 필요
        self.dummy_input_values = torch.zeros(self.model_max_length, dtype=self.input_values_dtype)
        self.speech_start_token = self.tokenizer("<Speech>", add_special_tokens=False)["input_ids"][0]
        self.speech_end_token = self.tokenizer("</Speech>", add_special_tokens=False)["input_ids"][0]

    def __call__(self, instances: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        import ipdb; ipdb.set_trace()
        input_values = [self.create_overlapping_window(instance["input_values"]) for instance in instances]
        assert self.sample_rate == instances[0]['sample_rate']

        input_features = [self.processor(
            audio=input_value,
            sampling_rate=self.sample_rate,
            padding=self.padding,
            # max_length=self.model_max_length,
            return_tenors="pt").input_features for input_value in input_values]
        
        input_features = np.stack(input_features, axis=0)
        input_features = torch.tensor(input_features, dtype=torch.float32)
    

        # For now, we only support Tokenizers with `padding_side = "right"` during Training (but plan to extend!)
        #   => Handle padding via RNN Utils => `pad_sequence`
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)

        # Truncate (if necessary)
        input_ids, labels = input_ids[:, : self.model_max_length], labels[:, : self.model_max_length]

        # Get `attention_mask` by checking for `pad_token_id`
        attention_mask = input_ids.ne(self.pad_token_id)
        # === Handle "unimodal" (language-only) vs. "multimodal" ===

        # Some examples are "language-only" --> build a Tensor of `multimodal_indices` that we can slice into easily
        multimodal_indices = torch.tensor(
            [idx for idx in range(len(input_values)) if input_values[idx] is not None], dtype=torch.long
        )
        return dict(
            input_values=input_features,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            multimodal_indices=multimodal_indices,
        )

    def create_overlapping_window(audio, window_size: int, overlap: int):
        step_size = window_size - overlap
        num_windows = (len(audio) - window_size) // step_size + 1
        n = audio.strides[0]
        return np.lib.stride_tricks.as_strided(audio, shape=(num_windows, window_size), strides=(step_size * n, n))



@dataclass
class LongSpeechPaddedCollatorForLanguageModeling:
    processor: WhisperProcessor
    tokenizer: PreTrainedTokenizerBase
    model_max_length: int
    padding: Union[bool, str]
    pad_token_id: int
    sample_rate: int = 16000
    padding_side: str = "right"
    input_values_dtype: torch.dtype = torch.float32

    def __post_init__(self) -> None:
        # TODO: 확인 필요
        self.dummy_input_values = torch.zeros(self.model_max_length, dtype=self.input_values_dtype)
        self.speech_start_token = self.tokenizer("<Speech>", add_special_tokens=False)["input_ids"][0]
        self.speech_end_token = self.tokenizer("</Speech>", add_special_tokens=False)["input_ids"][0]

    def __call__(self, instances: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_values = [instance["input_values"] for instance in instances]
        assert self.sample_rate == instances[0]['sample_rate']
        if isinstance(input_values[0], list):
            input_values = [np.stack(input_value, axis=1) for input_value in input_values]
            input_features = [[self.processor(audio=input_item, sampling_rate=self.sample_rate, padding=False, return_tensors="pt").input_features  for input_item in input_value] for input_value in input_values]
        import ipdb; ipdb.set_trace()
        num_sample = len(input_features[0])
        input_feature_processed = []
        for idx in range(num_sample):
            input_feature_temp = [input_feature[idx] for input_feature in input_features]
            input_feature_temp = torch.cat(input_feature_temp, axis=0)
            input_feature_processed.append(input_feature_temp)
          

        # For now, we only support Tokenizers with `padding_side = "right"` during Training (but plan to extend!)
        #   => Handle padding via RNN Utils => `pad_sequence`
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)

        # Truncate (if necessary)
        input_ids, labels = input_ids[:, : self.model_max_length], labels[:, : self.model_max_length]

        # Get `attention_mask` by checking for `pad_token_id`
        attention_mask = input_ids.ne(self.pad_token_id)
        # === Handle "unimodal" (language-only) vs. "multimodal" ===

        # Some examples are "language-only" --> build a Tensor of `multimodal_indices` that we can slice into easily
        multimodal_indices = torch.tensor(
            [idx for idx in range(len(input_values)) if input_values[idx] is not None], dtype=torch.long
        )
        return dict(
            input_values=input_feature_processed,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            multimodal_indices=multimodal_indices,
        )

    def _get_speech_tokens(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        speech_start_tokens = torch.full((batch_size, 1), self.speech_start_token, dtype=torch.long)
        speech_end_tokens = torch.full((batch_size, 1), self.speech_end_token, dtype=torch.long)
        return speech_start_tokens, speech_end_tokens
    