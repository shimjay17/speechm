"""
base_strategy.py

Abstract class definition of a (distributed) training strategy, with full annotations of class methods, utility
functions, and initialization logic.

Training Strategies (DDP, FSDP-Grad, FSDP-Full) tend to have a lot of repeated components; this class does a lot of
heavy lifting.
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler, SequentialSampler
from tqdm import tqdm
from transformers.modeling_outputs import CausalLMOutputWithPast

from cobra.models.slms import CobraSLM
from cobra.overwatch import initialize_overwatch
from cobra.training.metrics import Metrics
from cobra.util import check_bloat16_supported
from cobra.util.batching_utils import SplitModalitySampler
from cobra.util.data_utils import PaddedCollatorForLanguageModeling, SpeechPaddedCollatorForLanguageModeling, GigaSpeechPaddedCollatorForLanguageModeling

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


# === Abstract Base Class for an arbitrary Training Strategy ===
class TrainingStrategy(ABC):
    def __init__(
        self,
        slm: CobraSLM,
        device_id: int,
        epochs: int,
        max_steps: Optional[int],
        global_batch_size: int,
        per_device_batch_size: int,
        learning_rate: float,
        weight_decay: float,
        max_grad_norm: float,
        lr_scheduler_type: str,
        warmup_ratio: float,
        enable_gradient_checkpointing: bool = True,
        enable_mixed_precision_training: bool = True,
        reduce_in_full_precision: bool = False,
        mixed_precision_dtype: torch.dtype = torch.bfloat16,
        worker_init_fn: Optional[Callable[[int], None]] = None,
        save_checkpoint_step: int = 1000,
        **_: str,
    ) -> None:
        self.slm, self.device_id = slm, device_id

        # Get relevant SLM instance parameters before they get (potentially) wrapped
        self.all_module_keys, self.trainable_module_keys = self.slm.all_module_keys, self.slm.trainable_module_keys
        self.llm_transformer_layer_cls = self.slm.llm_backbone.transformer_layer_cls

        # Optimization Parameters
        self.epochs, self.max_steps = epochs, max_steps
        self.global_batch_size, self.per_device_batch_size = global_batch_size, per_device_batch_size

        self.learning_rate, self.weight_decay, self.max_grad_norm = learning_rate, weight_decay, max_grad_norm
        self.lr_scheduler_type, self.warmup_ratio = lr_scheduler_type, warmup_ratio

        # Generic Strategy Parameters
        self.enable_gradient_checkpointing = enable_gradient_checkpointing
        self.enable_mixed_precision_training = enable_mixed_precision_training
        self.reduce_in_full_precision = reduce_in_full_precision
        self.mixed_precision_dtype = mixed_precision_dtype

        # DataLoader Parameters
        self.worker_init_fn = worker_init_fn

        # Optimizers & Scheduler (initialized in `run_setup`)
        self.optimizer, self.lr_scheduler = None, None
        self.save_checkpoint_step = save_checkpoint_step

        # Lightweight Validation
        assert (
            self.global_batch_size % self.per_device_batch_size == 0
        ), "Per-Device Batch Size must evenly divide Global Batch Size!"
        self.grad_accumulation_steps = self.global_batch_size // self.per_device_batch_size // overwatch.world_size()
        if self.enable_mixed_precision_training:
            assert self.mixed_precision_dtype == torch.bfloat16, "Only BF16 mixed precision training is supported!"
            assert check_bloat16_supported(), "BFloat16 is not supported on this hardware; unset `mixed_precision`"

    @abstractmethod
    def save_checkpoint(
        self,
        run_dir: Path,
        global_step: int,
        epoch: int,
        train_loss: Optional[float] = None,
        only_trainable: bool = True,
    ) -> None: ...

    @abstractmethod
    def save_checkpoint_and_optimizer(
        self,
        run_dir: Path,
        global_step: int,
        epoch: int,
        train_loss: Optional[float] = None,
        only_trainable: bool = True,
    ) -> None: ...

    @abstractmethod
    def resume_from_checkpoint(self, checkpoint_path: Path) -> None: ...

    @abstractmethod
    def run_setup(self, run_dir: Path, n_train_examples: int) -> None: ...

    @abstractmethod
    def clip_grad_norm(self) -> None: ...

    def run_training(
        self,
        libri_dataset: Optional[Dataset],
        giga_dataset: Optional[Dataset],
        eval_dataset: Dataset,
        libri_collator: Optional[SpeechPaddedCollatorForLanguageModeling],
        giga_collator: Optional[GigaSpeechPaddedCollatorForLanguageModeling],
        metrics: Metrics,
        stage: str = "finetune",
        batch_construction_strategy: str = "split-modality",
        seed: int = 7,
    ) -> None:
        """Run the training loop for the given `dataset` and `collator`; log losses, results to `metrics`"""
        if "finetune" in stage and batch_construction_strategy == "split-modality":
            # Instantiate the split-modality sampler; if you want to extend with other batch construction schemes,
            #   (e.g., grouping by length) =>> can easily add them here!
            modality_lengths = libri_dataset.get_modality_lengths()
            sampler = SplitModalitySampler(
                libri_dataset,
                modality_lengths,
                global_batch_size=self.global_batch_size,
                num_replicas=overwatch.world_size(),
                rank=overwatch.rank(),
                seed=seed,
                drop_last=False,
            )

        else:
            
            libri_sampler = DistributedSampler(
                libri_dataset,
                num_replicas=overwatch.world_size(),
                rank=overwatch.rank(),
                shuffle=True,
                seed=seed,
                drop_last=False,
            ) if libri_dataset else None
            
            giga_sampler = DistributedSampler(
                giga_dataset,
                num_replicas=overwatch.world_size(),
                rank=overwatch.rank(),
                shuffle=True,
                seed=seed,
                drop_last=False,
            ) if giga_dataset else None

        eval_sampler = SequentialSampler(eval_dataset)

        dynamic_batch_size = self.per_device_batch_size // 2 if giga_dataset and libri_dataset else self.per_device_batch_size

        libri_dataloader = DataLoader(
            libri_dataset,
            batch_size=dynamic_batch_size,
            sampler=libri_sampler,
            collate_fn=libri_collator,
            num_workers=0,
            worker_init_fn=self.worker_init_fn,
        ) if libri_dataset else None

        giga_dataloader = DataLoader(
            giga_dataset,
            batch_size=dynamic_batch_size,
            sampler=giga_sampler,
            collate_fn=giga_collator,
            num_workers=0,
            worker_init_fn=self.worker_init_fn,
        ) if giga_dataset else None

        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=self.per_device_batch_size,
            sampler=eval_sampler,
            collate_fn=giga_collator,
            num_workers=2,
            # num_workers=6, # added by esyoon 2024-06-10-23:33:40 for debug
            worker_init_fn=self.worker_init_fn,
        )


        # print(self.grad_accumulation_steps)
        # Max Steps vs. Epochs Computation
        steps_per_epoch = (len(libri_dataloader) + len(giga_dataloader)) // self.grad_accumulation_steps if libri_dataset and giga_dataset else \
                        len(libri_dataloader) // self.grad_accumulation_steps if libri_dataset else \
                        len(giga_dataloader) // self.grad_accumulation_steps

        if self.max_steps is not None and steps_per_epoch < self.max_steps:
            self.epochs = 100

        status = metrics.get_status()
        with tqdm(
            total=(
                (self.epochs * steps_per_epoch)
                if self.max_steps is None
                else self.max_steps
            ),
            desc=status,
            leave=False,
            disable=not overwatch.is_rank_zero(),
        ) as progress:
            train_step = 0
            for epoch in range(self.epochs):
                self.slm.train()
                if libri_dataset:
                    libri_sampler.set_epoch(epoch)
                if giga_dataset:
                    giga_sampler.set_epoch(epoch)

                self.optimizer.zero_grad()

                libri_iterator = iter(libri_dataloader) if libri_dataset else None
                giga_iterator = iter(giga_dataloader) if giga_dataset else None

                for train_idx in range(steps_per_epoch * self.grad_accumulation_steps):
                    if libri_dataset:
                        libri_batch = next(libri_iterator, None)
                    if giga_dataset:
                        giga_batch = next(giga_iterator, None)

                    combined_loss = 0
                    if libri_batch is not None:
                        with torch.autocast(
                            "cuda",
                            dtype=self.mixed_precision_dtype,
                            enabled=self.enable_mixed_precision_training,
                        ):
                            libri_output: CausalLMOutputWithPast = self.slm(
                                input_ids=libri_batch["input_ids"],
                                attention_mask=libri_batch["attention_mask"],
                                input_values=libri_batch["input_values"],
                                labels=libri_batch["labels"],
                                multimodal_indices=libri_batch["multimodal_indices"],
                            )
                            libri_loss = libri_output.loss
                            libri_batch_size = libri_batch['input_ids'].shape[0]
                        combined_loss += libri_loss * libri_batch_size / self.per_device_batch_size
                        metrics.commit(libri_loss=libri_loss)

                    if giga_batch is not None:
                        with torch.autocast(
                            "cuda",
                            dtype=self.mixed_precision_dtype,
                            enabled=self.enable_mixed_precision_training,
                        ):
                            giga_output: CausalLMOutputWithPast = self.slm(
                                input_ids=giga_batch["input_ids"],
                                attention_mask=giga_batch["attention_mask"],
                                input_values=giga_batch["input_values"],
                                labels=giga_batch["labels"],
                                multimodal_indices=giga_batch["multimodal_indices"],
                            )
                            giga_loss = giga_output.loss
                            giga_batch_size = giga_batch['input_ids'].shape[0]
                        combined_loss += giga_loss * giga_batch_size / self.per_device_batch_size
                        metrics.commit(giga_loss=giga_loss)

                    normalized_loss = combined_loss / self.grad_accumulation_steps
                    normalized_loss.backward()

                    metrics.commit(loss=combined_loss)

                    if (train_idx + 1) % self.grad_accumulation_steps == 0:
                        metrics.commit(update_step_time=True)

                        self.clip_grad_norm()

                        self.optimizer.step()
                        self.lr_scheduler.step()
                        self.optimizer.zero_grad()

                        if metrics.global_step % 20 == 0:
                            metrics.eval = True
                            eval_loss = 0.0
                            eval_step = 0
                            for eval_batch in eval_dataloader:
                                with torch.no_grad():
                                    output: CausalLMOutputWithPast = self.slm(
                                        input_ids=eval_batch["input_ids"],
                                        attention_mask=eval_batch["attention_mask"],
                                        input_values=eval_batch["input_values"],
                                        labels=eval_batch["labels"],
                                        multimodal_indices=eval_batch["multimodal_indices"],
                                    )
                                    eval_loss += output.loss
                                    eval_step += 1
                            eval_loss /= eval_step
                            metrics.commit(eval_loss=eval_loss)
                        metrics.eval = False

                        metrics.commit(global_step=metrics.global_step + 1, lr=self.lr_scheduler.get_last_lr()[0])

                        status = metrics.push()

                        if self.max_steps is not None and metrics.global_step >= self.max_steps:
                            self.save_checkpoint(metrics.run_dir, metrics.global_step, epoch, combined_loss.item())
                            dist.barrier()
                            return

                        progress.update()
                        progress.set_description(status)

                        train_step += 1

                    if train_step % self.save_checkpoint_step == 0:
                        self.save_checkpoint_and_optimizer(metrics.run_dir, metrics.global_step, epoch, combined_loss.item())

                if self.max_steps is None:
                    self.save_checkpoint(metrics.run_dir, metrics.global_step, epoch, combined_loss.item())
                    dist.barrier()
