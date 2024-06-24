"""
materialize.py

Factory class for initializing pretraining datasets on a per-VLM basis; provides and exports individual functions for
clear control flow.
"""
from typing import Tuple, Type

from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from cobra.conf import DatasetConfig
from cobra.models.backbones.llm.prompting import PromptBuilder
from cobra.models.backbones.vision import ImageTransform
from cobra.preprocessing_speech.datasets_speech import AlignDataset, FinetuneDataset, AlignGigaDataset # added by esyoon 2024-06-09-20:40:46
from cobra.util.speech_data_utils import SpeechPaddedCollatorForLanguageModeling, GigaSpeechPaddedCollatorForLanguageModeling

# Dataset Initializers =>> Maps Stage --> cls()
DATASET_INITIALIZER = {"align": AlignDataset, "finetune": FinetuneDataset, "full-finetune": FinetuneDataset, "align_eval": AlignDataset, 'align_giga': AlignGigaDataset}


def get_dataset_and_collator(
    stage: str,
    dataset_cfg: DatasetConfig,
    tokenizer: PreTrainedTokenizerBase,
    prompt_builder_fn: Type[PromptBuilder],
    sample_rate:int = 16000,
    padding_side: str = "right",
    processor = None,
    padding='max_length',
) -> Tuple: # [Dataset, Collator]
    dataset_cls = DATASET_INITIALIZER[stage]
    dataset_root_dir = dataset_cfg.dataset_root_dir

    collator = SpeechPaddedCollatorForLanguageModeling(
        processor, tokenizer, tokenizer.model_max_length, padding, tokenizer.pad_token_id, sample_rate, padding_side=padding_side
    )

    # Switch on `stage`
    if stage == "align":
        data_json = dataset_cfg.align_stage_components
        if dataset_cfg.dataset_id == 'gigaspeech':
            dataset_cls = DATASET_INITIALIZER[f"{stage}_giga"]
            collator = GigaSpeechPaddedCollatorForLanguageModeling(
                processor, tokenizer, tokenizer.model_max_length, padding, tokenizer.pad_token_id, sample_rate, padding_side=padding_side
            )

        dataset = dataset_cls(
            data_json, tokenizer, prompt_builder_fn=prompt_builder_fn
        )
        eval_data_json = dataset_cfg.align_stage_eval_components
        
        eval_dataset = None
        if eval_data_json is not None:
            eval_dataset = dataset_cls(
                eval_data_json, tokenizer
            )
        return dataset, eval_dataset, collator

    elif stage == "finetune":
        annotation_json, image_dir = dataset_cfg.finetune_stage_components
        dataset = dataset_cls(
            dataset_root_dir / annotation_json,
            dataset_root_dir / image_dir,
            tokenizer,
            prompt_builder_fn=prompt_builder_fn,
        )
        return dataset, collator

    elif stage == "align_eval":
        annotation_json, image_dir = dataset_cfg.finetune_stage_components
        dataset = dataset_cls(
            dataset_root_dir / annotation_json,
            dataset_root_dir / image_dir,
            tokenizer,
            prompt_builder_fn=prompt_builder_fn,
        )
        return dataset, collator


    elif stage == "full-finetune":
        annotation_json, image_dir = dataset_cfg.finetune_stage_components
        dataset = dataset_cls(
            dataset_root_dir / annotation_json,
            dataset_root_dir / image_dir,
            image_transform,
            tokenizer,
            prompt_builder_fn=prompt_builder_fn,
        )
        return dataset, collator

    else:
        raise ValueError(f"Stage `{stage}` is not supported!")
