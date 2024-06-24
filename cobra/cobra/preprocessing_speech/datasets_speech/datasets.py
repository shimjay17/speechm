"""
datasets.py

PyTorch Dataset Definitions for Cobra models; supports processing for both the `align` and `finetune` stages, with
utilities for formatting conversations during the `finetune` stage subject to the given LLM backbone's expected
formatting (e.g., SYS_PROMPT + USER: ... ASSISTANT: ... for Vicuña v1.5 Chat models).

We currently only support Map-style Datasets; assumes that all files (annotations, images) are on local disk, and that
random access image reading is relatively cheap/fast.
"""
import copy
import json
from pathlib import Path
import soundfile as sf
from typing import Dict, List, Tuple, Type
import numpy as np
import random
import librosa

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase, GPTNeoXTokenizerFast

from cobra.models.backbones.llm.prompting import PromptBuilder
from cobra.models.backbones.vision import ImageTransform

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100


class AlignDataset(Dataset[Dict[str, torch.Tensor]]):
    def __init__(
        self,
        data_json: Path,
        tokenizer: PreTrainedTokenizerBase,
        prompt_builder_fn: Type[PromptBuilder] = None,

    ) -> None:
        super().__init__()
        self.data_json = data_json
        self.tokenizer = tokenizer
        self.dataset_type = "align"

        # Create Prompt Template or template list
        self.prompt_template = "{text}" + self.tokenizer.eos_token

        # Load Chat JSON
        with open(self.data_json, "r") as f:
            examples_dict = json.load(f)
        
    
        # for now use all of the dataset of librispeech for the align stage
        self.examples = []
        for key in examples_dict.keys():
            for example in examples_dict[key]:
                self.examples.append(example)   

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Following the *actual* code executed from the LLaVa codebase, during the "align" phase, we actually discard
        the "prompt" from the human, and instead directly predict the caption from the image.

        As a concrete example given the "raw data" for the first example:
            example = self.examples[0]["conversations]` = {
                [
                    {"from": "human", "value": "Render a clear and concise summary of the photo.\n<image>"},
                    {"from": "gpt", "value": "select luxury furniture 3 - inch gel memory foam mattress topper"}
                ]
            }

        Return =>> self.tokenizer("<image> select luxury furniture 3 - inch gel memory foam mattress topper\n")

        :param idx: Index to retrieve from the dataset.

        :return: Dictionary of {"pixel_values": torch.Tensor, "input_ids": torch.Tensor, "labels": torch.Tensor}
        """
        
        audio_path, text = Path(self.examples[idx]["path"]), self.examples[idx]["text"]
        # assert (len(conversation) == 2) and ("<image>" not in conversation[-1]["value"]), "Unexpected text!"

        # Format Caption --> {caption}{eos_token}
        transcription = self.prompt_template.format(text=text.strip())

        # We treat image patches as "tokens = [p1 p2 p3, ...]"; we need to specify ordering of text/patch tokens.
        #   => Critically, we find that inserting *after* the BOS token leads to the strongest performance!
        #       - input_ids = "<s> p1 p2 p3 ... <caption_text> \n"
        #       - labels = "IGNORE IGNORE ..." (copy `input_ids` replacing <s> and p{1...K} with IGNORE)
        #
        # IMPORTANT => IF WE'RE USING HF LLM.forward(... labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
        input_ids = self.tokenizer(transcription, truncation=True, return_tensors="pt").input_ids[0]
        labels = copy.deepcopy(input_ids)

        # For tokenizers that have the <BOS> token: 
        # Set the <BOS> token's label to IGNORE_INDEX (since we're inserting the image patches right after)
        # Mamba/GPTNeoXTokenizer does not have the <BOS> token.
        if not isinstance(self.tokenizer, GPTNeoXTokenizerFast):
            labels[0] = IGNORE_INDEX

        # Process Image --> get "pixel_values" (will either be a torch.Tensor OR a Dict[str,torch.Tensor])
        input_values, sample_rate = sf.read(audio_path)
        return dict(input_values=input_values, input_ids=input_ids, labels=labels, sample_rate=sample_rate)


    def __len__(self) -> int:
        return len(self.examples)

class AlignPromptDataset(Dataset[Dict[str, torch.Tensor]]):
    def __init__(
        self,
        data_json: Path,
        tokenizer: PreTrainedTokenizerBase,
        prompt_builder_fn: Type[PromptBuilder] = None,
        max_duration: int = 180 # 3 minutes 

    ) -> None:
        super().__init__()
        self.data_json = data_json
        self.tokenizer = tokenizer
        self.dataset_type = "align"
        self.prompt_json = "/data2/data_account/workspace/samsung/speech_summarization_mamba/cobra/data/prompt/prompt.json"
        self.max_duration = max_duration

        # Create Prompt Template or template list
        with open(self.prompt_json, 'r') as f:
            prompt_dict = json.load(f)
        self.prompt_list = prompt_dict['asr']

        # Load Chat JSON
        with open(self.data_json, "r") as f:
            examples_dict = json.load(f)
        
    
        # for now use all of the dataset of librispeech for the align stage
        # we use librispeech 960h + giga speech
        self.examples = []
        for key in examples_dict.keys():
            for example in examples_dict[key]:
                self.examples.append(example)

    # def _merge_segments(self, )  

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Following the *actual* code executed from the LLaVa codebase, during the "align" phase, we actually discard
        the "prompt" from the human, and instead directly predict the caption from the image.

        As a concrete example given the "raw data" for the first example:
            example = self.examples[0]["conversations]` = {
                [
                    {"from": "human", "value": "Render a clear and concise summary of the photo.\n<image>"},
                    {"from": "gpt", "value": "select luxury furniture 3 - inch gel memory foam mattress topper"}
                ]
            }

        Return =>> self.tokenizer("<image> select luxury furniture 3 - inch gel memory foam mattress topper\n")

        :param idx: Index to retrieve from the dataset.

        :return: Dictionary of {"pixel_values": torch.Tensor, "input_ids": torch.Tensor, "labels": torch.Tensor}
        """
        
        audio_path, text = Path(self.examples[idx]["path"]), self.examples[idx]["text"]
        # assert (len(conversation) == 2) and ("<image>" not in conversation[-1]["value"]), "Unexpected text!"

        # Format Caption --> {caption}{eos_token}
        transcription = self.prompt_template.format(text=text.strip())

        # We treat image patches as "tokens = [p1 p2 p3, ...]"; we need to specify ordering of text/patch tokens.
        #   => Critically, we find that inserting *after* the BOS token leads to the strongest performance!
        #       - input_ids = "<s> p1 p2 p3 ... <caption_text> \n"
        #       - labels = "IGNORE IGNORE ..." (copy `input_ids` replacing <s> and p{1...K} with IGNORE)
        #
        # IMPORTANT => IF WE'RE USING HF LLM.forward(... labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
        input_ids = self.tokenizer(transcription, truncation=True, return_tensors="pt").input_ids[0]
        labels = copy.deepcopy(input_ids)

        # For tokenizers that have the <BOS> token: 
        # Set the <BOS> token's label to IGNORE_INDEX (since we're inserting the image patches right after)
        # Mamba/GPTNeoXTokenizer does not have the <BOS> token.
        if not isinstance(self.tokenizer, GPTNeoXTokenizerFast):
            labels[0] = IGNORE_INDEX

        # Process Image --> get "pixel_values" (will either be a torch.Tensor OR a Dict[str,torch.Tensor])
        input_values, sample_rate = sf.read(audio_path)
        # pixel_values = self.image_transform(Image.open(self.image_dir / image_path).convert("RGB"))
        return dict(input_values=input_values, input_ids=input_ids, labels=labels, sample_rate=sample_rate)

    # def get_modality_lengths(self, n_image_patches: int) -> List[Tuple[bool, int]]: # 이거는 아마 쓸모 없을 것 같은데 왜냐면 speech는 length가 길어서
    #     """Get a list of modalities (unimodal / text-only vs. multimodal) and length of conversations per example."""
    #     modality_lengths = []
    #     for example in self.examples:
    #         is_multimodal = "path" in example
    #         n_words = sum(len(example["text"]))
    #         modality_lengths.append((is_multimodal, (n_image_patches + n_words) if is_multimodal else n_words))
    #     return modality_lengths

    def __len__(self) -> int:
        return len(self.examples)

class AlignGigaDataset(Dataset[Dict[str, torch.Tensor]]):
    def __init__(
        self,
        data_json: Path,
        tokenizer: PreTrainedTokenizerBase,
        prompt_builder_fn: Type[PromptBuilder] = None,
        max_duration: int = 180,  # 3 minutes

    ) -> None:
        super().__init__()
        self.data_json = data_json
        self.tokenizer = tokenizer
        self.dataset_type = "align"
        self.prompt_builder_fn = prompt_builder_fn
        self.instruction_json = "/data2/data_account/workspace/samsung/speech_summarization_mamba/cobra/data/prompt/prompt.json"
        # Create Prompt Template or template list

        # Load Chat JSON
        with open(self.data_json, "r") as f:
            examples_dict = json.load(f)

        # Load instruction Jsin
        with open(self.instruction_json, "r") as f:
            instruction_dict = json.load(f)
        
        self.instruction_list = instruction_dict['asr']
    
        # for now use all of the dataset of librispeech for the align stage
        self.examples = []
        for key in examples_dict.keys():
            for example in examples_dict[key]:
                self.examples.append(example)   
    
    def _generate_segment(self, audio, timesteps:list, sample_rate:int = 16000):
        segment = [audio[int(start * sample_rate): int(end*sample_rate)] for (start, end) in timesteps]
        return np.concatenate(segment)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Following the *actual* code executed from the LLaVa codebase, during the "align" phase, we actually discard
        the "prompt" from the human, and instead directly predict the caption from the image.

        As a concrete example given the "raw data" for the first example:
            example = self.examples[0]["conversations]` = {
                [
                    {"from": "human", "value": "Render a clear and concise summary of the photo.\n<image>"},
                    {"from": "gpt", "value": "select luxury furniture 3 - inch gel memory foam mattress topper"}
                ]
            }

        Return =>> self.tokenizer("<image> select luxury furniture 3 - inch gel memory foam mattress topper\n")

        :param idx: Index to retrieve from the dataset.

        :return: Dictionary of {"pixel_values": torch.Tensor, "input_ids": torch.Tensor, "labels": torch.Tensor}
        """

        
        audio_path, text = Path(self.examples[idx]["path"].replace('opus', 'wav')), self.examples[idx]["text"]
        input_values, sample_rate = sf.read(audio_path)

        input_values = self._generate_segment(input_values, self.examples[idx]["segment_time"], sample_rate)

        duration = self.examples[idx]["segment_duration"]

        text = " ".join(text)
        insturciton_idx = random.randint(0, len(self.instruction_list)-1)
        instruction = self.instruction_list[insturciton_idx]

        prompt_builder, input_ids, labels = self.prompt_builder_fn(model_family="cobra"), [], []
        conversation =[
                {"from": "human", "value": instruction},
                {"from": "gpt", "value": text}
            ]
        for turn_idx, turn in enumerate(conversation):
            # Get "effective" string added to prompt --> handle whitespace for tokenizer type!
            msg = prompt_builder.add_turn(turn["from"], turn["value"])

            if isinstance(self.tokenizer, GPTNeoXTokenizerFast):
                pass
            else:
                raise ValueError(f"Tokenizer of type `{type(self.tokenizer)}` is not explicitly handled!")

            # Tokenize Input IDs
            turn_input_ids = self.tokenizer(msg, add_special_tokens=turn_idx == 0).input_ids

            # [CRITICAL] We do not want to take the loss for the "USER: <msg>" prompts =>> just the responses!
            turn_labels = (
                [IGNORE_INDEX for _ in range(len(turn_input_ids))] if (turn_idx % 2) == 0 else list(turn_input_ids)
            )

            # Add to Trackers
            input_ids.extend(turn_input_ids)
            labels.extend(turn_labels)
        # Tensorize =>> Set the <BOS> token's label to IGNORE_INDEX (since we're inserting the image patches after)
        #   - IMPORTANT => IF WE'RE USING HF LLM.forward(... labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)

        # Handle Truncation (if necessary)
        input_ids, labels = input_ids[: self.tokenizer.model_max_length], labels[: self.tokenizer.model_max_length]

        # We treat image patches as "tokens = [p1 p2 p3, ...]"; we need to specify ordering of text/patch tokens.
        #   => Critically, we find that inserting *after* the BOS token leads to the strongest performance!
        #       - input_ids = "<s> p1 p2 p3 ... <caption_text> \n"
        #       - labels = "IGNORE IGNORE ..." (copy `input_ids` replacing <s> and p{1...K} with IGNORE)
        #
        # IMPORTANT => IF WE'RE USING HF LLM.forward(... labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
        # For tokenizers that have the <BOS> token: 
        # Set the <BOS> token's label to IGNORE_INDEX (since we're inserting the image patches right after)
        # Mamba/GPTNeoXTokenizer does not have the <BOS> token.
        if not isinstance(self.tokenizer, GPTNeoXTokenizerFast):
            labels[0] = IGNORE_INDEX

        return dict(input_values=input_values, input_ids=input_ids, labels=labels, sample_rate=sample_rate, duration=duration)


    def __len__(self) -> int:
        return len(self.examples)

class FinetuneDataset(Dataset[Dict[str, torch.Tensor]]):
    def __init__(
        self,
        instruct_json: Path,
        image_dir: Path,
        image_transform: ImageTransform,
        tokenizer: PreTrainedTokenizerBase,
        prompt_builder_fn: Type[PromptBuilder],
    ) -> None:
        super().__init__()
        self.instruct_json, self.image_dir = instruct_json, image_dir
        self.image_transform, self.tokenizer = image_transform, tokenizer
        self.prompt_builder_fn = prompt_builder_fn
        self.dataset_type = "finetune"

        # Load Instruct JSON
        with open(self.instruct_json, "r") as f:
            self.examples = json.load(f)

    # === Unimodal + Multimodal Handling ===
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Unlike the *align* stage handling, for the *finetune* stage, we actually need to handle multiple "turns" of
        dialog grounded in a single image.

        To do this, we leverage the `prompt_builder_fn` which instantiates a PromptBuilder object. By calling the
        methods for adding turns and getting a prompt, we ensure proper formatting and consistency for each example.

        :param idx: Index to retrieve from the dataset.

        :return: Dictionary of {"pixel_values": torch.Tensor, "input_ids": torch.Tensor, "labels": torch.Tensor}
        """
        conversation = self.examples[idx]["conversations"]

        # Create Prompt Builder --> add each message sequentially
        prompt_builder, input_ids, labels = self.prompt_builder_fn(model_family="cobra"), [], []
        for turn_idx, turn in enumerate(conversation):
            # Get "effective" string added to prompt --> handle whitespace for tokenizer type!
            msg = prompt_builder.add_turn(turn["from"], turn["value"])

            if isinstance(self.tokenizer, GPTNeoXTokenizerFast):
                pass
            else:
                raise ValueError(f"Tokenizer of type `{type(self.tokenizer)}` is not explicitly handled!")

            # Tokenize Input IDs
            turn_input_ids = self.tokenizer(msg, add_special_tokens=turn_idx == 0).input_ids

            # [CRITICAL] We do not want to take the loss for the "USER: <msg>" prompts =>> just the responses!
            turn_labels = (
                [IGNORE_INDEX for _ in range(len(turn_input_ids))] if (turn_idx % 2) == 0 else list(turn_input_ids)
            )

            # Add to Trackers
            input_ids.extend(turn_input_ids)
            labels.extend(turn_labels)

        # Tensorize =>> Set the <BOS> token's label to IGNORE_INDEX (since we're inserting the image patches after)
        #   - IMPORTANT => IF WE'RE USING HF LLM.forward(... labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)

        # Handle Truncation (if necessary)
        input_ids, labels = input_ids[: self.tokenizer.model_max_length], labels[: self.tokenizer.model_max_length]

        # === Handle "unimodal" (language-only) vs. "multimodal" ===
        if "image" in self.examples[idx]:
            image_path = Path(self.examples[idx]["image"])

            # Set the <BOS> token's label to IGNORE_INDEX (since we're inserting the image patches right after)
            # Mamba/GPTNeoXTokenizer does not have the <BOS> token.
            if not isinstance(self.tokenizer, GPTNeoXTokenizerFast):
                labels[0] = IGNORE_INDEX

            # Process Image --> get "pixel_values" (will either be a torch.Tensor OR a Dict[str,torch.Tensor])
            pixel_values = self.image_transform(Image.open(self.image_dir / image_path).convert("RGB"))

            return dict(pixel_values=pixel_values, input_ids=input_ids, labels=labels)

        else:
            # No image --> return `pixel_values` = None; Collator will do the smart batch handling for us!
            return dict(pixel_values=None, input_ids=input_ids, labels=labels)

    def get_modality_lengths(self) -> List[Tuple[bool, int]]:
        """Get a list of modalities (unimodal / text-only vs. multimodal) and length of conversations per example."""
        modality_lengths = []
        for example in self.examples:
            is_multimodal = "image" in example
            n_words = sum([len(turn["value"].split()) for turn in example["conversations"]])
            modality_lengths.append((is_multimodal, n_words))
        return modality_lengths

    def __len__(self) -> int:
        return len(self.examples)
