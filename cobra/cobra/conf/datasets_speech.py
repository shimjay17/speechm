"""
datasets.py

Draccus Dataclass Definition for a DatasetConfig object, with various registered subclasses for each dataset variant
and processing scheme. A given dataset variant (e.g., `llava-lightning`) configures the following attributes:
    - Dataset Variant (Identifier) --> e.g., "llava-v15"
    - Align Stage Dataset Components (annotations, images)
    - Finetune Stage Dataset Components (annotations, images)
    - Dataset Root Directory (Path)
"""
from dataclasses import dataclass
from enum import Enum, unique
from pathlib import Path
from typing import Tuple

from draccus import ChoiceRegistry


@dataclass
class DatasetConfig(ChoiceRegistry):
    # fmt: off
    dataset_id: str                                 # Unique ID that fully specifies a dataset variant

    # Dataset Components for each Stage in < align | finetune >
    align_stage_components: Path = None      # Path to annotation file containing text and audio file paths
    finetune_stage_components: Path = None  # Path to annotation file containing text and audio file paths for `finetune` stage
    align_stage_eval_components: Path = Path("data") 
    finetune_stage_eval_components: Path = None


    dataset1_config = None 
    dataset2_config = None

    # dataset_root_dir: Path                          # Path to dataset root directory; others paths are relative to root
    # fmt: on

# @dataclass
# class MultiDatasetConfig(ChoiceRegistry):
#     # fmt: off
#     dataset_id: str # Unique ID that fully specifies a dataset variant

#     # Dataset Components for each Stage in < align | finetune >
#     dataset1_config: DatasetConfig 
#     dataset2_config: DatasetConfig

#     # dataset_root_dir: Path                          # Path to dataset root directory; others paths are relative to root
#     # fmt: on

# [Reproduction] LLaVa-v15 (exact dataset used in all public LLaVa-v15 models)
@dataclass
class LLaVa_V15_Config(DatasetConfig):
    dataset_id: str = "llava-v15"

    align_stage_components: Tuple[Path, Path] = (
        Path("download/llava-laion-cc-sbu-558k/chat.json"),
        Path("download/llava-laion-cc-sbu-558k/"),
    )
    finetune_stage_components: Tuple[Path, Path] = (
        Path("download/llava-v1.5-instruct/llava_v1_5_mix665k.json"),
        Path("download/llava-v1.5-instruct/"),
    )
    dataset_root_dir: Path = Path("data")


# [Multimodal-Only] LLava-v15 WITHOUT the Language-Only ShareGPT Data (No Co-Training)
@dataclass
class LLaVa_Multimodal_Only_Config(DatasetConfig):
    dataset_id: str = "llava-multimodal"

    align_stage_components: Tuple[Path, Path] = (
        Path("download/llava-laion-cc-sbu-558k/chat.json"),
        Path("download/llava-laion-cc-sbu-558k/"),
    )
    finetune_stage_components: Tuple[Path, Path] = (
        Path("download/llava-v1.5-instruct/llava_v1_5_stripped625k.json"),
        Path("download/llava-v1.5-instruct/"),
    )
    dataset_root_dir: Path = Path("data")


# LLaVa-v15 + LVIS-Instruct-4V
@dataclass
class LLaVa_LVIS4V_Config(DatasetConfig):
    dataset_id: str = "llava-lvis4v"

    align_stage_components: Tuple[Path, Path] = (
        Path("download/llava-laion-cc-sbu-558k/chat.json"),
        Path("download/llava-laion-cc-sbu-558k/"),
    )
    finetune_stage_components: Tuple[Path, Path] = (
        Path("download/llava-v1.5-instruct/llava_v1_5_lvis4v_mix888k.json"),
        Path("download/llava-v1.5-instruct/"),
    )
    dataset_root_dir: Path = Path("data")


# LLaVa-v15 + LRV-Instruct
@dataclass
class LLaVa_LRV_Config(DatasetConfig):
    dataset_id: str = "llava-lrv"

    align_stage_components: Tuple[Path, Path] = (
        Path("download/llava-laion-cc-sbu-558k/chat.json"),
        Path("download/llava-laion-cc-sbu-558k/"),
    )
    finetune_stage_components: Tuple[Path, Path] = (
        Path("download/llava-v1.5-instruct/llava_v1_5_lrv_mix1008k.json"),
        Path("download/llava-v1.5-instruct/"),
    )


# LLaVa-v15 + LVIS-Instruct-4V + LRV-Instruct
@dataclass
class LLaVa_LVIS4V_LRV_Config(DatasetConfig):
    dataset_id: str = "llava-lvis4v-lrv"

    align_stage_components: Tuple[Path, Path] = (
        Path("download/llava-laion-cc-sbu-558k/chat.json"),
        Path("download/llava-laion-cc-sbu-558k/"),
    )
    finetune_stage_components: Tuple[Path, Path] = (
        Path("download/llava-v1.5-instruct/llava_v1_5_lvis4v_lrv_mix1231k.json"),
        Path("download/llava-v1.5-instruct/"),
    )
    dataset_root_dir: Path = Path("data")

# Librispeech # added by esyoon 2024-06-06-20:42:58
@dataclass
class Librispeech_Config(DatasetConfig):
    dataset_id: str = "librispeech"

    align_stage_components: Path =  Path("/data2/data_account/workspace/samsung/speech_summarization_mamba/cobra/data/librispeech/train.json")
    align_stage_eval_components: Path =  Path("/data2/data_account/workspace/samsung/speech_summarization_mamba/cobra/data/librispeech/test.json")

    finetune_stage_components: Path =  Path("/data2/data_account/workspace/samsung/speech_summarization_mamba/cobra/data/librispeech/train.json")

    dataset_root_dir: Path = Path("data")

@dataclass
class Gigaspeech_Config(DatasetConfig):
    dataset_id: str = "gigaspeech"

    align_stage_components: Path =  Path("/data2/data_account/workspace/samsung/speech_summarization_mamba/cobra/data/gigaspeech/train_ver3_3min.json")
    align_stage_eval_components = None

    finetune_stage_components: Path =  Path("/data2/data_account/workspace/samsung/speech_summarization_mamba/cobra/data/gigaspeech/train_ver3_3min.json")

    dataset_root_dir: Path = Path("data")

@dataclass
class Librispeech_Gigaspeech_config(DatasetConfig):
    dataset_id: str = "librispeech+gigaspeech"

    dataset1_config = Librispeech_Config
    dataset2_config = Gigaspeech_Config

@dataclass
class Mediasum_Config(DatasetConfig):
    dataset_id: str = "mediasum"

    finetune_stage_components: Path = Path("/data2/data_account/workspace/cobra/data/TTS/mediasum_speech/train_data/train.json")
    finetune_stage_eval_components: Path = Path("/data2/data_account/workspace/cobra/data/TTS/mediasum_speech/val_data/val.json")

    dataset_root_dir: Path = Path("data")

# === Define a Dataset Registry Enum for Reference & Validation =>> all *new* datasets must be added here! ===
@unique
class DatasetRegistry(Enum):
    # === LLaVa v1.5 ===
    LLAVA_V15 = LLaVa_V15_Config

    LLAVA_MULTIMODAL_ONLY = LLaVa_Multimodal_Only_Config

    LLAVA_LVIS4V = LLaVa_LVIS4V_Config
    LLAVA_LRV = LLaVa_LRV_Config

    LLAVA_LVIS4V_LRV = LLaVa_LVIS4V_LRV_Config

    # Librispeech
    LIBRISPEECH = Librispeech_Config

    GIGASPEECH = Gigaspeech_Config

    # Multiple Datasets
    LIBRISPEECH_GIGASPEECH = Librispeech_Gigaspeech_config

    MEDIASUM = Mediasum_Config


    @property
    def dataset_id(self) -> str:
        return self.value.dataset_id


# Register Datasets in Choice Registry
for dataset_variant in DatasetRegistry:
    DatasetConfig.register_subclass(dataset_variant.dataset_id, dataset_variant.value)
