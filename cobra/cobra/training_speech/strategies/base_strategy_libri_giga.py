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
        libri_dataset: Dataset,
        giga_dataset: Dataset,
        eval_dataset: Dataset,
        libri_collator: SpeechPaddedCollatorForLanguageModeling,
        giga_collator: GigaSpeechPaddedCollatorForLanguageModeling,
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
            modality_lengths += giga_dataset.get_modality_lengths()
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
            )
            giga_sampler = DistributedSampler(
                giga_dataset,
                num_replicas=overwatch.world_size(),
                rank=overwatch.rank(),
                shuffle=True,
                seed=seed,
                drop_last=False,
            )
 
        eval_sampler = SequentialSampler(eval_dataset) # libri

        # Create a DataLoader with the initialized sampler, per-device-bsz, and collator
        # libri_dataloader = DataLoader(
        #     libri_dataset,
        #     batch_size=self.per_device_batch_size // 2,
        #     sampler=libri_sampler,
        #     collate_fn=libri_collator,
        #     # num_workers=2,
        #     num_workers=0, # added by esyoon 2024-06-10-23:33:40 for debug
        #     worker_init_fn=self.worker_init_fn,
        # )
        giga_dataloader = DataLoader(
            giga_dataset,
            batch_size=self.per_device_batch_size // 2,
            sampler=giga_sampler,
            collate_fn=giga_collator,
            # num_workers=0,
            num_workers=6, # added by esyoon 2024-06-10-23:33:40 for debug
            worker_init_fn=self.worker_init_fn,
        )
        libri_dataloader = DataLoader(
            libri_dataset,
            batch_size=self.per_device_batch_size // 2,
            sampler=libri_sampler,
            collate_fn=libri_collator,
            # num_workers=0,
            num_workers=6, # added by esyoon 2024-06-10-23:33:40 for debug
            worker_init_fn=self.worker_init_fn,
        )

        giga_iterator = iter(giga_dataloader)


        # Create a evaluation dataloader with the initialized sampler, per-device-bsz, and collator
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=self.per_device_batch_size,
            sampler=eval_sampler,
            collate_fn=libri_collator,
            # num_workers=2,
            num_workers=6, # added by esyoon 2024-06-10-23:33:40 for debug
            worker_init_fn=self.worker_init_fn,
        )
        # print(self.grad_accumulation_steps)
        # Max Steps vs. Epochs Computation
        steps_per_epoch = len(libri_dataloader) // self.grad_accumulation_steps
        if self.max_steps is not None and steps_per_epoch < self.max_steps:
            # Just set `epochs` to some large number --> we'll short-circuit based on steps anyway
            self.epochs = 100

        # === Train ===
        status = metrics.get_status()
        with tqdm(
            total=(
                (self.epochs * len(libri_dataloader) // self.grad_accumulation_steps)
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
                libri_sampler.set_epoch(epoch)
                giga_sampler.set_epoch(epoch)

                # Zero-Gradients (just in case)
                self.optimizer.zero_grad()

                # Note that we'll unpack batch (and let AMP/FSDP do its thing) in the SLM.forward() call
                #   => Basically, if we're using mixed precision (or not), autocast()/FSDP will move to device!
                for train_idx, libri_batch in enumerate(libri_dataloader):
                    # Get Giga Batch
                    try:
                        giga_batch = next(giga_iterator)
                    except:
                        giga_iterator = iter(giga_dataloader)
                        giga_batch = next(giga_iterator)
                    # Combine Libri & Giga Batches
                    # batch = {k: torch.cat([libri_batch[k], giga_batch[k]], dim=0) for k in libri_batch.keys()}
                    # [Contract] self.slm.forward() must automatically compute `loss` and return!
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
                        giga_batch_size = giga_batch['input_ids'].shape[0]

                    libri_loss = (libri_loss * libri_batch_size)/(libri_batch_size + giga_batch_size) #added by HSY 6/22/24
                    metrics.commit(libri_loss=libri_loss) #added by HSY 6/22/24
                    normalized_loss = libri_loss / self.grad_accumulation_steps
                    normalized_loss.backward()

                    with torch.autocast(
                        "cuda",
                        dtype=self.mixed_precision_dtype,
                        enabled=self.enable_mixed_precision_training,
                    ):                      
                        #torch.cuda.empty_cache()
                        giga_output: CausalLMOutputWithPast = self.slm(
                            input_ids=giga_batch["input_ids"],
                            attention_mask=giga_batch["attention_mask"],
                            input_values=giga_batch["input_values"],
                            labels=giga_batch["labels"],
                            multimodal_indices=giga_batch["multimodal_indices"],
                            num_segments=giga_batch['num_segments']
                        )
                        giga_loss = giga_output.loss
                        
                        
                    giga_loss = (giga_loss * giga_batch_size)/(libri_batch_size + giga_batch_size)
                    metrics.commit(giga_loss=giga_loss) #added by HSY 6/22/24
                    normalized_loss = giga_loss / self.grad_accumulation_steps
                    
                    normalized_loss.backward()                        


                    loss = libri_loss + giga_loss
                    # Commit Loss (Prior to Gradient Accumulation Normalization)
                    metrics.commit(loss=loss)

                    # Normalize Loss to account for Gradient Accumulation --> Backward!
                    # [IMPORTANT] Technically speaking, doing gradient accumulation in this way is "incorrect"; this is
                    #             because in general, each batch has a *different number of masked out tokens* (because
                    #             we're instruct-tuning). Taking the mean over two unbalanced means != the right thing!
                    #
                    #             HOWEVER -- at least at the 7B scale, the "naive" approach is just as performant as
                    #             the "correct" implementation, without adding extra complexity.
                    #
                    # That being said =>> at the 13B scale, *no matter what we tried, ANY gradient accumulation is just
                    #   really bad for downstream performance. Initial investigation shows that BF16 accumulation
                    #   just really tanks in precision... and don't have a good/clean way to fix this. Would love for
                    #   someone to PR and fix this (and I'd greatly appreciate it!!!)
                    # normalized_loss = loss / self.grad_accumulation_steps
                    # normalized_loss.backward()

                    # Step =>> Only if Done w/ Gradient Accumulation
                    if (train_idx + 1) % self.grad_accumulation_steps == 0:
                        metrics.commit(update_step_time=True)

                        # Clip Gradients --> this is custom, per-strategy because of DDP vs. FSDP locality-assumptions
                        self.clip_grad_norm()

                        # Optimizer & LR Scheduler Step
                        self.optimizer.step()
                        self.lr_scheduler.step()
                        self.optimizer.zero_grad()

                    if metrics.global_step % 20 == 0:
                        metrics.eval = True
                        eval_loss = 0.0
                        eval_step = 0
                        for eval_idx, eval_batch in enumerate(eval_dataloader):
                            with torch.no_grad():
                                # self.slm.generate(prompt_text='<|endoftext|>', input_values=eval_batch["input_values"][0], 
                                #                   use_cache=False, max_length=1024, temperature=0.9, top_p=0.95,
                                #                   )
                                # import ipdb; ipdb.set_trace()
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
                        
                                

                    # update the status and progress bar
                    if (train_idx + 1) % self.grad_accumulation_steps == 0:
                        # metrics.commit(update_step_time=True)

                        # # Clip Gradients --> this is custom, per-strategy because of DDP vs. FSDP locality-assumptions
                        # self.clip_grad_norm()

                        # # Optimizer & LR Scheduler Step
                        # self.optimizer.step()
                        # self.lr_scheduler.step()
                        # self.optimizer.zero_grad()

                        # Push Metrics
                        metrics.commit(global_step=metrics.global_step + 1, lr=self.lr_scheduler.get_last_lr()[0])
                        status = metrics.push()

                        # Check for Termination & Save Final Checkpoint (in case `max_steps` is not None)
                        if self.max_steps is not None and metrics.global_step >= self.max_steps:
                            self.save_checkpoint(metrics.run_dir, metrics.global_step, epoch, loss.item())
                            dist.barrier()

                            return

                        # Update Progress Bar
                        progress.update()
                        progress.set_description(status)

                        train_step += 1
                    metrics.eval = False
                    
                    if train_step % self.save_checkpoint_step == 0:
                        self.save_checkpoint_and_optimizer(metrics.run_dir, metrics.global_step, epoch, loss.item())

                        # added by esyoon 2024-06-14-19:34:07
                        # do evaludation after saving checkpoint
                    

            # Save checkpoint at end each epoch (if `self.max_steps` is None)
            if self.max_steps is None:
                self.save_checkpoint(metrics.run_dir, metrics.global_step, epoch, loss.item())
                dist.barrier()
