"""
pretrain.py

Pretraining script for Cobra VLM pretraining in native PyTorch, using Fully-Sharded Data Parallel (FSDP) to run
distributed training across GPUs. By default, assumes that CUDA toolkit is >= 11.0 (to support BF16 mixed precision).


Notes & Prerequisites:
    - If you want to set a custom location for all HF / TIMM artifacts --> `export HF_HOME="<PATH>"` *before* running!
        => For example (add to end of .bashrc): `export HF_HOME="/mnt/fsx/skaramcheti/cache"`

Run with:
    - [Single Node One-GPU (Debug)] : torchrun --standalone --nnodes 1 --nproc-per-node 1 scripts/pretrain.py
    - [Single Node Multi-GPU (= $K)]: torchrun --standalone --nnodes 1 --nproc-per-node $K scripts/pretrain.py
    - [Multi-Node/AWS Sagemaker] Depends on your individual setup; file an issue if you have trouble!
"""
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, Union

import draccus
import torch
import torch.distributed as dist
import yaml
import shutil

from cobra.conf import DatasetConfig, DatasetRegistry, ModelConfig, ModelRegistry
from cobra.models import get_llm_backbone_and_tokenizer, get_speech_backbone, get_slm
from cobra.overwatch import initialize_overwatch
# from cobra.preprocessing import get_dataset_and_collator
from cobra.preprocessing_speech import get_dataset_and_collator
from cobra.training_speech import Metrics, get_train_strategy
from cobra.util import set_global_seed

# Disable Tokenizers Parallelism to Play Nice w/ PyTorch Multiprocessing DataLoaders
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# for debug turn off wandb
# os.environ['WANDB_MODE'] = 'disabled'

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


@dataclass
class PretrainConfig:
    # fmt: off

    # ModelConfig (`cobra/conf/models.py`); override with --model.type `ModelRegistry.<MODEL>.model_id`
    model: ModelConfig = field(
        default_factory=ModelConfig.get_choice_class(ModelRegistry.COBRA_3B.model_id)
    )

    # DatasetConfig (`cobra/conf/datasets.py`); override with --dataset.type `DatasetRegistry.<DATASET>.dataset_id`
    dataset: DatasetConfig = field(
        default_factory=DatasetConfig.get_choice_class(DatasetRegistry.LLAVA_V15.dataset_id)
    )

    # Pretraining Stage in < align (projector-only) | finetune (projector + LLM) | full-finetune (all) >
    # ---
    stage: str = "finetune"                                         # Pretraining Stage in < align | finetune >
    pretrained_checkpoint: Optional[Path] = None                    # Pretrained Checkpoint to Load (for `finetune`)
                                                                    #   if None =>> will match on (run_dir / `align`)

    # Run Arguments
    run_id: Optional[str] = None                                    # Run ID for logging, Weights & Biases
    run_root_dir: Path = Path("runs")                               # Path to directory to store logs & checkpoints
    seed: int = 7                                                   # Random seed (for reproducibility)

    # HF Hub Credentials (for any gated models)
    hf_token: Union[str, Path] = Path(".hf_token")                  # Environment variable or Path to HF Token

    # Tracking Parameters
    trackers: Tuple[str, ...] = ("jsonl", "wandb")                  # Trackers to initialize (if W&B, add config!)
    wandb_project: str = "samsung_saummarization"                                # Name of W&B project (default: `cobra`)
    wandb_entity: Optional[str] = None                              # Name of W&B entity (default: None)

    def __post_init__(self) -> None:
        """Set optimization parameters based on `stage` in {"align", "finetune"}."""
        if self.stage == "align":
            self.epochs = self.model.align_epochs
            self.max_steps = self.model.align_max_steps
            self.global_batch_size = self.model.align_global_batch_size
            self.per_device_batch_size = self.model.align_per_device_batch_size

            self.learning_rate = self.model.align_learning_rate
            self.weight_decay = self.model.align_weight_decay
            self.max_grad_norm = self.model.align_max_grad_norm
            self.lr_scheduler_type = self.model.align_lr_scheduler_type
            self.warmup_ratio = self.model.align_warmup_ratio

            self.train_strategy = self.model.align_train_strategy

        elif self.stage.endswith("finetune"):
            self.epochs = self.model.finetune_epochs
            self.max_steps = self.model.finetune_max_steps
            self.global_batch_size = self.model.finetune_global_batch_size
            self.per_device_batch_size = self.model.finetune_per_device_batch_size

            self.learning_rate = self.model.finetune_learning_rate
            self.weight_decay = self.model.finetune_weight_decay
            self.max_grad_norm = self.model.finetune_max_grad_norm
            self.lr_scheduler_type = self.model.finetune_lr_scheduler_type
            self.warmup_ratio = self.model.finetune_warmup_ratio

            self.train_strategy = self.model.finetune_train_strategy

        else:
            raise ValueError(f"Stage `{self.stage}` is not supported!")

    # fmt: on


@draccus.wrap()
def pretrain(cfg: PretrainConfig) -> None:
    overwatch.info("Cobra VLM Training :: Gathering Light")

    # Note => Under `torchrun` initializing `overwatch` will automatically set up `torch.distributed`
    torch.cuda.set_device(device_id := (overwatch.rank() % torch.cuda.device_count()))
    torch.cuda.empty_cache()

    # Create Unique Run Name & Save Directory
    model_id = cfg.model.model_id
    dataset_id = cfg.dataset.dataset_id
    # if (dataset_id := cfg.dataset.dataset_id) == "llava-v15":
    #     cfg.run_id = f"{model_id}+stage-{cfg.stage}+x{cfg.seed}" if cfg.run_id is None else cfg.run_id
    # else:
    #     cfg.run_id = f"{dataset_id}+{model_id}+stage-{cfg.stage}+x{cfg.seed}" if cfg.run_id is None else cfg.run_id
    cfg.run_id = f"{dataset_id}+{model_id}+stage-{cfg.stage}+x{cfg.seed}" if cfg.run_id is None else cfg.run_id

    # Start =>> Build Directories and Set Randomness
    hf_token = cfg.hf_token.read_text().strip() if isinstance(cfg.hf_token, Path) else os.environ[cfg.hf_token]
    worker_init_fn = set_global_seed(cfg.seed, get_worker_init_fn=True)
    os.makedirs(run_dir := (cfg.run_root_dir / cfg.run_id), exist_ok=True)
    os.makedirs(cfg.run_root_dir / cfg.run_id / "checkpoints", exist_ok=True)
    if overwatch.is_rank_zero():
        # Additionally save a JSON version of the config
        draccus.dump(cfg, open(run_dir / "config.yaml", "w"))
        with open(run_dir / "config.yaml", "r") as f_yaml, open(run_dir / "config.json", "w") as f_json:
            yaml_cfg = yaml.safe_load(f_yaml)
            json.dump(yaml_cfg, f_json, indent=2)

    # Also save the nn_utils.py
    shutil.copyfile('cobra/util/nn_utils.py', run_dir / 'nn_utils.py')

    # Load Vision Backbone --> on CPU, in Full Precision (initializing model, image_transform via TIMM)
    # overwatch.info(f"Loading Vision Backbone [bold]{cfg.model.vision_backbone_id}[/] via TIMM ")
    # vision_backbone, image_transform = get_vision_backbone_and_transform(
    #     cfg.model.vision_backbone_id, image_resize_strategy=cfg.model.image_resize_strategy
    # )
    overwatch.info(f"Loading Speech Backbone [bold]{cfg.model.speech_backbone_id}[/] via HF ")
    speech_backbone, speech_processor = get_speech_backbone(
        cfg.model.speech_backbone_id, cfg.model.speech_backbone_path, speech_augmentation_strategy=cfg.model.speech_augmentation_strategy
    )
    # Load LLM Backbone --> on CPU, in Full Precision (initializing Tokenizer + handling special tokens if necessary)
    overwatch.info(f"Loading Pretrained LLM [bold]{cfg.model.llm_backbone_id}[/] via HF Transformers")
    llm_backbone, tokenizer = get_llm_backbone_and_tokenizer(
        cfg.model.llm_backbone_id, llm_max_length=cfg.model.llm_max_length, hf_token=hf_token, llm_backbone_path=cfg.model.llm_backbone_path
    )

    # Create VLM => wraps `vision_backbone` and `llm`
    overwatch.info(f"Instantiating CobraVLM `{model_id}` for Training Stage = `{cfg.stage}`")
    # vlm = get_vlm(
    #     model_id,
    #     cfg.model.arch_specifier,
    #     vision_backbone,
    #     llm_backbone,
    #     enable_mixed_precision_training=cfg.model.enable_mixed_precision_training,
    # )
    slm = get_slm(
        model_id,
        cfg.model.arch_specifier,
        speech_backbone,
        llm_backbone,
        second_per_window=cfg.model.second_per_window,
        second_stride=cfg.model.second_stride,
        enable_mixed_precision_training=cfg.model.enable_mixed_precision_training,
    )

    total_params = sum(p.numel() for p in slm.parameters())
    train_params = sum(p.numel() for p in slm.parameters() if p.requires_grad)
    proj_params = sum(p.numel() for p in slm.projector.parameters())
    overwatch.info(f"Total Parameters = `{total_params / 1e9 :.2f}B`")
    overwatch.info(f"Trainable Parameters = `{train_params / 1e9 :.2f}B`")
    overwatch.info(f"Projector Parameters = `{proj_params / 1e6:.2f}M`")

    # [Explicit] Call to `freeze_backbones` here for clarity => will log exactly what is frozen / what's not!
    overwatch.info(f"Invoking `SLM.freeze_backbones()` for `{model_id}` => Training Stage: `{cfg.stage}`")
    slm.freeze_backbones(cfg.stage)

    # Load Weights from Checkpoint (depends on stage, config)
    overwatch.info(f"Invoking `VLM.load_checkpoint()` for `{model_id}` => Training Stage: `{cfg.stage}`")
    slm.load_from_checkpoint(cfg.stage, run_dir, pretrained_checkpoint=cfg.pretrained_checkpoint)

    # Get Dataset for Specified Stage
    overwatch.info(f"Creating Dataset `{cfg.dataset.dataset_id}` => Stage: `{cfg.stage}`")
    libri_config = cfg.dataset.dataset1_config
    gigaspeech_config = cfg.dataset.dataset2_config
    libri_train_dataset, libri_eval_dataset, libri_collator = get_dataset_and_collator(
        cfg.stage,
        libri_config,
        tokenizer=tokenizer,
        prompt_builder_fn=llm_backbone.prompt_builder_fn,
        processor=speech_processor,
        padding_side=tokenizer.padding_side,
    )

    gigaspeech_train_dataset, _, gigaspeech_collator = get_dataset_and_collator( # eval_dataset is None currently # added by esyoon 2024-06-21-02:36:32
        cfg.stage,
        gigaspeech_config,
        tokenizer=tokenizer,
        prompt_builder_fn=llm_backbone.prompt_builder_fn,
        processor=speech_processor,
        padding_side=tokenizer.padding_side,
    )
    #

    # for fast evaluation
    eval_dataset = torch.utils.data.Subset(libri_eval_dataset, range(1000))

    # Create Train Strategy
    overwatch.info(f"Initializing Train Strategy `{cfg.train_strategy}`")
    train_strategy = get_train_strategy(
        train_strategy=cfg.train_strategy,
        slm=slm,
        device_id=device_id,
        epochs=cfg.epochs,
        max_steps=cfg.max_steps,
        global_batch_size=cfg.global_batch_size,
        per_device_batch_size=cfg.per_device_batch_size,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        max_grad_norm=cfg.max_grad_norm,
        lr_scheduler_type=cfg.lr_scheduler_type,
        warmup_ratio=cfg.warmup_ratio,
        enable_gradient_checkpointing=cfg.model.enable_gradient_checkpointing,
        enable_mixed_precision_training=cfg.model.enable_mixed_precision_training,
        reduce_in_full_precision=cfg.model.reduce_in_full_precision,
        worker_init_fn=worker_init_fn,
    )
    train_strategy.run_setup(run_dir=run_dir, n_train_examples=len(gigaspeech_train_dataset)+len(libri_train_dataset)) # TODO: 둘다 받을 수 있게 수정

    # Create Metrics =>> Handles on the fly tracking, logging to specified trackers (e.g., JSONL, Weights & Biases)
    overwatch.info(f"Creating Metrics with Active Trackers => `{cfg.trackers}`")

    if cfg.dataset.dataset_id == "librispeech+gigaspeech": #added by HSY 6/22/24
        metrics = Metrics(
            cfg.trackers,
            cfg.run_id,
            run_dir,
            draccus.encode(cfg),
            cfg.stage,
            wandb_project=cfg.wandb_project,
            wandb_entity=cfg.wandb_entity,
            grad_accumulation_steps=train_strategy.grad_accumulation_steps,
            giga_speech=True
        )
    else:
        metrics = Metrics(
            cfg.trackers,
            cfg.run_id,
            run_dir,
            draccus.encode(cfg),
            cfg.stage,
            wandb_project=cfg.wandb_project,
            wandb_entity=cfg.wandb_entity,
            grad_accumulation_steps=train_strategy.grad_accumulation_steps,
        )

    # Run Training
    overwatch.info("Starting Training Loop")
    train_strategy.run_training(libri_dataset=libri_train_dataset, giga_dataset=gigaspeech_train_dataset, eval_dataset=eval_dataset, libri_collator=libri_collator, giga_collator=gigaspeech_collator, metrics=metrics, stage=cfg.stage, seed=cfg.seed)

    # Finalize
    overwatch.info("Done with Training =>> Finalizing Metrics")
    metrics.finalize()

    # And... we're done!
    overwatch.info("... and that's all, folks!")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    pretrain()
