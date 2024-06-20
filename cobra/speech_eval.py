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

from cobra.conf import DatasetConfig, DatasetRegistry, ModelConfig, ModelRegistry
from cobra.models import load_slm, get_llm_backbone_and_tokenizer, get_speech_backbone, get_slm
from cobra.overwatch import initialize_overwatch
from cobra.preprocessing_speech import get_dataset_and_collator
from cobra.training_speech import Metrics, get_train_strategy
from cobra.util import set_global_seed


# Disable Tokenizers Parallelism to Play Nice w/ PyTorch Multiprocessing DataLoaders
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# for debug turn off wandb
os.environ['WANDB_MODE'] = 'disabled'

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)

@dataclass
class EvaluationConfig:
    # fmt: off

    # DatasetConfig from `vlm_eval/conf/datasets.py`; override with --dataset.type `DatasetRegistry.<DATASET>.dataset_id`
    # dataset: DatasetConfig = field(
    #     default_factory=DatasetConfig.get_choice_class(DatasetRegistry.POPE_FULL.dataset_id)
    # )
    
    model_family: str = "mamba"
    model_id: str = "test"
    # model_dir: Optional[Path] = None                # Path to model checkpoint to load --> should be self-contained
    # model_dir: str = "/data2/mamba/workspace/neurips24/cobra/runs/cobra+3b+stage-finetune+x7"
    model_dir: str = "/mnt/hdd/hsyoon/workspace/samsung/cobra/runs/qmamba+align"

    speech_backbone_id: str = "whisper-large-v2"
    speech_backbone_path: str = "/mnt/hdd/hsyoon/workspace/models/whisper-large-v2"
    llm_backbone_id: str = "mamba-2.8b-zephyr"
    llm_backbone_path: str = "/mnt/hdd/hsyoon/workspace/models/mamba-2.8b-zephyr"

    # Inference Parameters
    device_batch_size: int = 1                      # Device Batch Size set to 1 until LLaVa/HF LLaMa fixes bugs!
    num_workers: int = 2                            # Number of Dataloader Workers (on each process)

    # Artifact Parameters
    results_dir: Path = Path(                       # Path to results directory (writing predicted output, metrics)
        "/mnt/hdd/jay/workspace/vlmevaluation/scripts/results"
    )

    root_dir: Path = Path("/mnt/hdd/jay/workspace/vlmevaluation")

    # HF Hub Credentials (for LLaMa-2)
    hf_token: Union[str, Path] = Path("/mnt/hdd/hsyoon/workspace/samsung/cobra/.hf_token")  # Environment variable or Path to HF Token

    # Randomness
    seed: int = 21                                  # Random Seed (for reproducibility)

    def __post_init__(self) -> None:
        self.run_dir = self.model_dir

    # fmt: on

def encode_audio(audio_file_path: str, processor: WhisperProcessor, sample_rate: int = 16000) -> Dict[str, torch.Tensor]:
    # Load audio file
    waveform, sr = torchaudio.load(audio_file_path)
    assert sr == sample_rate, f"Expected sample rate {sample_rate}, but got {sr}"
    
    # Process input values
    input_values = waveform.squeeze().numpy()  # Convert to numpy array if necessary
    input_features = processor(
        audio=input_values,
        sampling_rate=sample_rate,
        return_tensors="pt"
    ).input_features
    
    # input_features = torch.tensor(input_features, dtype=torch.float32)
    
    return input_features


@draccus.wrap()
def evaluate(cfg: EvaluationConfig) -> None:
    
    audio_file_path = "/mnt/hdd/hsyoon/workspace/ES/speech/datasets/LibriSpeech/dev-clean/1272/128104/1272-128104-0000.flac"
    prompt_text = " "

    hf_token = Path(cfg.hf_token).read_text().strip()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    
    slm = load_slm(
        cfg.model_family,
        cfg.model_id,
        cfg.run_dir,
        hf_token=hf_token
    )

    prompt_text = " "


    prompt_builder = slm.get_prompt_builder()
    prompt_builder.add_turn(role="human", message=prompt_text)
    prompt_text_raw = prompt_builder.get_prompt()

    prompt_input = (prompt_text_raw,)

    # Load Whisper processor
    processor = WhisperProcessor.from_pretrained(cfg.speech_backbone_path)

    result = encode_audio(audio_file_path, processor)  # Using the processor to encode audio
    
    generated_text = slm.generate_answer(
        result,
        prompt_text,
        return_string_probabilities=None,
    )

    #assert if the generated text is ' ' 

    print(generated_text)

if __name__ == "__main__":
    evaluate()
