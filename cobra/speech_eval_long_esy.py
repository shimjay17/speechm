import os
import json
import yaml
import shutil
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Union
import copy

import draccus
import torch
import torch.distributed as dist
import torchaudio
from torch.nn.utils.rnn import pad_sequence
from transformers import WhisperProcessor, PreTrainedTokenizerBase
from accelerate import Accelerator, InitProcessGroupKwargs
from torch.utils.data import DataLoader
from datetime import timedelta
from tqdm import tqdm

from cobra.conf import DatasetConfig, DatasetRegistry, ModelConfig, ModelRegistry
from cobra.models import load_slm, get_llm_backbone_and_tokenizer, get_speech_backbone, get_slm
from cobra.overwatch import initialize_overwatch
from cobra.preprocessing_speech import get_dataset_and_collator
from cobra.training_speech import Metrics, get_train_strategy
from cobra.util import set_global_seed

from cobra.util.speech_data_utils import LongSpeechPaddedCollatorForLanguageModeling


from jiwer import wer  # Import WER calculation function
import pandas as pd  # Import pandas for saving results to Excel
import soundfile as sf  # Import soundfile for reading audio files

# Disable Tokenizers Parallelism to Play Nice w/ PyTorch Multiprocessing DataLoaders
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# for debug turn off wandb
os.environ['WANDB_MODE'] = 'disabled'

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)

class AlignEvalDataset(torch.utils.data.Dataset[Dict[str, torch.Tensor]]):
    def __init__(self, data_json: Path, tokenizer: PreTrainedTokenizerBase, dataset_name= None, samples_per_item:int =2) -> None:
        super().__init__()
        self.data_json = data_json
        self.tokenizer = tokenizer
        self.dataset_type = "align"

        # Create Prompt Template
        self.prompt_template = "{text}" + self.tokenizer.eos_token
        self.IGNORE_INDEX = -100
        self.samples_per_item = samples_per_item

        # Load Chat JSON
        with open(self.data_json, "r") as f:
            examples_dict = json.load(f)
        
        # for now use all of the dataset of librispeech for the align stage
        self.examples = []
        samples = []
        for idx, example in enumerate(examples_dict[dataset_name]):
            samples.append(example)
            if len(samples) == self.samples_per_item:
                self.examples.append(samples)
                samples = []
            else:
                pass

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        audio_path_list, text_list = [Path(sample["path"]) for sample in self.examples[idx]], [sample["text"] for sample in self.examples[idx]]
        transcription = [text.strip() for text in text_list]
        transcription = " ".join(transcription)
        transcription = self.prompt_template.format(text=transcription)
        input_ids = self.tokenizer(transcription, truncation=True, return_tensors="pt").input_ids[0]
        labels = copy.deepcopy(input_ids)
        audio_files = [sf.read(audio_path) for audio_path in audio_path_list]
        input_values = [audio_file[0] for audio_file in audio_files]
        sample_rate = audio_files[0][1]
        return dict(input_values=input_values, labels=labels, input_ids=input_ids, sample_rate=sample_rate)
    
    def __len__(self) -> int:
        return len(self.examples)

@dataclass
class EvaluationConfig:
    model_family: str = "mamba"
    model_id: str = "speech_align2"
    model_dir: str = "/mnt/hdd/hsyoon/workspace/samsung/cobra/runs/qmamba+align"
    speech_backbone_id: str = "whisper-large-v2"
    speech_backbone_path: str = "/mnt/hdd/hsyoon/workspace/models/whisper-large-v2"
    llm_backbone_id: str = "mamba-2.8b-zephyr"
    llm_backbone_path: str = "/mnt/hdd/hsyoon/workspace/models/mamba-2.8b-zephyr"
    device_batch_size: int = 1
    num_workers: int = 2
    results_dir: Path = Path("/mnt/hdd/jay/workspace/vlmevaluation/scripts/results")
    root_dir: Path = Path("/mnt/hdd/hsyoon/workspace/ES/speech/datasets/")  # Point to the directory containing the datasets
    hf_token: Union[str, Path] = Path("/mnt/hdd/hsyoon/workspace/samsung/cobra/.hf_token")
    seed: int = 21

    def __post_init__(self) -> None:
        self.run_dir = self.model_dir

def encode_audio(waveform: torch.Tensor, processor: WhisperProcessor, sample_rate: int = 16000) -> Dict[str, torch.Tensor]:

    input_values = waveform.squeeze().cpu().numpy()  # Convert to numpy array if necessary
    input_features = processor(
        audio=input_values,
        sampling_rate=sample_rate,
        return_tensors="pt"
    ).input_features

    input_features = input_features.to("cuda")

    return input_features

@draccus.wrap()
def evaluate(cfg: EvaluationConfig) -> None:
    hf_token = Path(cfg.hf_token).read_text().strip()

    # Initialize Accelerator
    accel = Accelerator(kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(seconds=18000))])
    device = accel.device
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    slm = load_slm(
        cfg.model_family,
        cfg.model_id,
        cfg.run_dir,
        hf_token=hf_token
    )

    prompt_builder = slm.get_prompt_builder()
    processor = WhisperProcessor.from_pretrained(cfg.speech_backbone_path)


    eval_data_json = Path("/mnt/hdd/hsyoon/workspace/samsung/cobra/cobra/data/libirspeech/test.json")

    eval_dataset_clean = AlignEvalDataset(eval_data_json, slm.tokenizer, 'test-clean')
    eval_dataset_other = AlignEvalDataset(eval_data_json, slm.tokenizer, 'test-other')
    collator = LongSpeechPaddedCollatorForLanguageModeling(
        processor, slm.tokenizer, slm.tokenizer.model_max_length, padding="max_length", pad_token_id=slm.tokenizer.pad_token_id
    )


    all_predictions = []
    all_references = []
    results = []
    eval_datalodaer_clean= DataLoader(eval_dataset_clean, batch_size=cfg.device_batch_size, num_workers=cfg.num_workers, pin_memory=True, collate_fn=collator)
    eval_dataloader_other = DataLoader(eval_dataset_other, batch_size=cfg.device_batch_size, num_workers=cfg.num_workers, pin_memory=True, collate_fn=collator)
    # for dataset, dataset_name in [(dataset_clean, "dev-clean"), (dataset_other, "dev-other")]:        # Initialize DataLoader
    # for dataset, dataset_name in [(dataset_clean, "train-clean-100"), (dataset_other, "train-other-500")]:        # Initialize DataLoader
    #     data_loader = DataLoader(dataset, batch_size=cfg.device_batch_size, num_workers=cfg.num_workers, pin_memory=True)

        # Prepare DataLoader with Accelerator
    eval_datalodaer_clean = accel.prepare(eval_datalodaer_clean)
    eval_dataloader_other = accel.prepare(eval_dataloader_other)
    
    #clean
    for i, batch in enumerate(tqdm(eval_datalodaer_clean, disable=not accel.is_local_main_process)):
        input_values = batch['input_values']
        # input_values= batch["input_values"]
        labels = batch["labels"].to(device, non_blocking=True)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        prompt_text = "."
        prompt_builder = slm.get_prompt_builder()
        prompt_builder.add_turn(role="human", message=prompt_text)
        prompt_text_raw = prompt_builder.get_prompt()

        prompt_input = (prompt_text_raw,)
        generated_text = slm.generate_answer(
            input_values,
            prompt_text,
            return_string_probabilities=None,
        )

        all_predictions.append(generated_text[0].upper())
        utterance = slm.tokenizer.decode(labels[0], skip_special_tokens=True).strip()
        all_references.append(utterance.upper())
        wer_temp = wer([utterance.upper()], [generated_text[0].upper()])
        import ipdb; ipdb.set_trace()

        results.append({
            "dataset": 'test-clean',  # Add dataset name to results
            "input": utterance,
            "generated": generated_text,
            "input_length": len(utterance[0].split()),
            "generated_length": len(generated_text[0].split()),
            "wer": wer_temp
        })

    # Save results to Excel only on the main process
    if accel.is_local_main_process:
        word_error_rate = wer(all_references, all_predictions)
        print(f"Word Error Rate (WER): {word_error_rate}")

        results_df = pd.DataFrame(results)
        results_file = cfg.results_dir / f"evaluation_results_{cfg.model_id}_clean.xlsx"
        results_df.to_excel(results_file, index=False)
        print(f"Results saved to {results_file}")
    
    # =============================================================================================================
    for i, batch in enumerate(tqdm(eval_dataloader_other, disable=not accel.is_local_main_process)):
        input_values = batch["input_values"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        prompt_text = " "
        prompt_builder = slm.get_prompt_builder()
        prompt_builder.add_turn(role="human", message=prompt_text)
        prompt_text_raw = prompt_builder.get_prompt()

        prompt_input = (prompt_text_raw,)
        
        generated_text = slm.generate_answer(
            input_values,
            prompt_text,
            return_string_probabilities=None,
        )

        all_predictions.append(generated_text[0].upper())
        utterance = slm.tokenizer.decode(labels[0], skip_special_tokens=True).strip()
        all_references.append(utterance.upper())
        wer_temp = wer([utterance.upper()], [generated_text[0].upper()])

        results.append({
            "dataset": 'test-other',  # Add dataset name to results
            "input": utterance,
            "generated": generated_text,
            "input_length": len(utterance[0].split()),
            "generated_length": len(generated_text[0].split()),
            "wer": wer_temp
        })

    # Save results to Excel only on the main process
    if accel.is_local_main_process:
        word_error_rate = wer(all_references, all_predictions)
        print(f"Word Error Rate (WER): {word_error_rate}")

        results_df = pd.DataFrame(results)
        results_file = cfg.results_dir / f"evaluation_results_{cfg.model_id}_other.xlsx"
        results_df.to_excel(results_file, index=False)

if __name__ == "__main__":
    evaluate()
