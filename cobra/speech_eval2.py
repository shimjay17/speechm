import os
import json
import yaml
import shutil
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Union

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

from jiwer import wer  # Import WER calculation function
import pandas as pd  # Import pandas for saving results to Excel

# Disable Tokenizers Parallelism to Play Nice w/ PyTorch Multiprocessing DataLoaders
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# for debug turn off wandb
os.environ['WANDB_MODE'] = 'disabled'

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)

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

    dataset_clean = torchaudio.datasets.LIBRISPEECH(
        root=cfg.root_dir,
        url="dev-clean",
        download=False
    )

    dataset_other = torchaudio.datasets.LIBRISPEECH(
        root=cfg.root_dir,
        url="dev-other",
        download=False
    )

    all_predictions = []
    all_references = []
    results = []

    for dataset, dataset_name in [(dataset_clean, "dev-clean"), (dataset_other, "dev-other")]:        # Initialize DataLoader
        data_loader = DataLoader(dataset, batch_size=cfg.device_batch_size, num_workers=cfg.num_workers, pin_memory=True)

        # Prepare DataLoader with Accelerator
        data_loader = accel.prepare(data_loader)

        for i, (waveform, sample_rate, utterance, _, _, _) in enumerate(tqdm(data_loader, disable=not accel.is_local_main_process)):
            result = encode_audio(waveform, processor, sample_rate)  # Modify encode_audio to accept waveform directly

            prompt_text = " "

            prompt_builder = slm.get_prompt_builder()
            prompt_builder.add_turn(role="human", message=prompt_text)
            prompt_text_raw = prompt_builder.get_prompt()

            prompt_input = (prompt_text_raw,)
            
            generated_text = slm.generate_answer(
                result,
                prompt_text,
                return_string_probabilities=None,
            )

            all_predictions.append(generated_text[0].upper())
            all_references.append(utterance[0].upper())

            results.append({
                "dataset": dataset_name,  # Add dataset name to results
                "input": utterance,
                "generated": generated_text,
                "input_length": len(utterance[0].split()),
                "generated_length": len(generated_text[0].split())
            })

    # Save results to Excel only on the main process
    if accel.is_local_main_process:
        word_error_rate = wer(all_references, all_predictions)
        print(f"Word Error Rate (WER): {word_error_rate}")

        results_df = pd.DataFrame(results)
        results_file = cfg.results_dir / f"evaluation_results_{cfg.model_id}.xlsx"
        results_df.to_excel(results_file, index=False)
        print(f"Results saved to {results_file}")

if __name__ == "__main__":
    evaluate()
