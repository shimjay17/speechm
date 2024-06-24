"""
cobra.py

PyTorch Module defining a CobraVLM, our general interface for defining the various different VLMs in our work.

Notes:
    - For now, we don't subclass `transformers.PretrainedModel` (or CausalLM). Instead, we assume a very limited subset
      of the {Model}ForCausalLM API that enables dispatch to the underlying LLM's `generate` utilities (feeding inputs
      through our custom projection shim).
"""
from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Callable, Dict, List, Optional, Type, Union
import os

import torch
from torch.distributed.fsdp.wrap import _module_wrap_policy, _or_policy
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import PreTrainedModel

from cobra.models.mamba.modeling_mamba_speech import MambaForCausalLM
from cobra.models.backbones.llm import LLMBackbone, MambaLLMBackbone
# from cobra.models.backbonesllm import LLaMa2LLMBackbone
from cobra.models.backbones.llm.prompting import PromptBuilder
from cobra.models.backbones.vision import VisionBackbone
from cobra.models.backbones.speech import HFSpeechBackbone
from cobra.models.slms.base_slm import SLM
from cobra.overwatch import initialize_overwatch
from cobra.util.nn_utils import FusedMLPProjector, LinearProjector, MLPProjector, FusedLDPProjector
from cobra.util.nn_utils import MambaProjectorV2, QMambaProjector, SimpleQMambaProjector, MambaProjectorV3
from cobra.models.mamba.modeling_mamba_speech import GenerationMixin as MambaGenerationMixin
# from cobra.models.materialize_speech import get_llm_backbone_and_tokenizer

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100


class CobraSLM(SLM):
    def __init__(
        self,
        model_id: str,
        speech_backbone: HFSpeechBackbone,
        llm_backbone: MambaLLMBackbone,
        second_per_window: float = 0.333333,
        second_stride: float = 0.333333,
        enable_mixed_precision_training: bool = True,
        arch_specifier: str = "gelu-mlp",
    ) -> None:
        super().__init__(
            "cobra",
            model_id,
            speech_backbone,
            llm_backbone,
            enable_mixed_precision_training=enable_mixed_precision_training,
        )

        # Set Weight Initialization Seed for Projector Consistency
        torch.manual_seed(speech_backbone.embed_dim)

        # Initialize Projection (Adapter) based on `arch_specifier`
        self.arch_specifier = arch_specifier 
        if arch_specifier == "linear":
            self.projector = LinearProjector(speech_backbone.embed_dim, llm_backbone.embed_dim)
        elif arch_specifier.endswith("fused-gelu-mlp"):
            self.projector = FusedMLPProjector(speech_backbone.embed_dim, llm_backbone.embed_dim)
        elif arch_specifier.endswith("gelu-mlp"):
            self.projector = MLPProjector(speech_backbone.embed_dim, llm_backbone.embed_dim)
        elif arch_specifier.endswith("fused-ldpnet"):
            self.projector = FusedLDPProjector(speech_backbone.embed_dim, llm_backbone.embed_dim)
        elif arch_specifier.endswith("qmamba"):
            self.projector = SimpleQMambaProjector(speech_backbone.embed_dim, llm_backbone.embed_dim, query_length=2)
        elif arch_specifier.endswith("vlmamba"): # added by esyoon 2024-06-09-21:52:31
            self.projector = MambaProjectorV2(speech_backbone.embed_dim, llm_backbone.embed_dim)
        elif arch_specifier.endswith("vlmamba-conv"):
            self.projector = MambaProjectorV3(speech_backbone.embed_dim, llm_backbone.embed_dim)
        else:
            raise ValueError(f"CobraSLM with `{arch_specifier = }` is not supported!")

        # Trackers
        self.speech_backbone_requires_grad = False

        # Set Module Keys =>> used in Checkpoint Saving / Model Loading
        self.all_module_keys = ["speech_backbone", "llm_backbone", "projector"]
        self.trainable_module_keys = []

        # === Generation Utilities ===
        #   => For computing likelihoods --> get tokens corresponding to "True", "False" and "Yes", "No"
        self.string2idx = {}
        for trigger_string in ["True", "False", "Yes", "No"] + [chr(ord("A") + i) for i in range(26)]:
            token_idx_list = self.llm_backbone.tokenizer.encode(trigger_string, add_special_tokens=False)
            assert len(token_idx_list) == 1, f'String "{trigger_string}" is tokenized as more than one token!'
            self.string2idx[trigger_string] = token_idx_list[0]

        self.eos_token_id = self.llm_backbone.tokenizer.eos_token_id

        self.second_per_window = second_per_window
        self.second_stride = second_stride

    def mamba_generate(self, *args, **kwargs):
        return MambaGenerationMixin.generate(self, *args, **kwargs)


    # @torch.inference_mode()
    # def generate(self, prompt_text: str, **kwargs: str) -> str:
    #     # For now, only support generation with a batch size of 1 for simplicity
    #     tokenizer = self.llm_backbone.tokenizer

    #     # Prepare Inputs
    #     input_ids = tokenizer(prompt_text, truncation=True, return_tensors="pt").input_ids.to(self.device)
    #     # input_values = image_transform(image)
    #     if isinstance(input_values, torch.Tensor):
    #         input_values = input_values[None, ...].to(self.device)
    #     elif isinstance(input_values, dict):
    #         input_values = {k: v[None, ...].to(self.device) for k, v in input_values.items()}
    #     else:
    #         raise ValueError(f"Unsupported `input_values` type = {type(input_values)}")

    #     # Invoke super().generate --> taps into `GenerationMixin` which (redirects) to `forward()`
    #     autocast_dtype = self.llm_backbone.half_precision_dtype
    #     with torch.autocast("cuda", dtype=autocast_dtype, enabled=self.enable_mixed_precision_training):
    #         # fmt: off
    #         import ipdb; ipdb.set_trace()
    #         generated_ids = self.mamba_generate(
    #             input_ids=input_ids,            # Shape: [1, seq]
    #             input_values=input_values,      # Shape: [1, 3, res, res] or Dict[str, Shape[1, 3, res, res]]
    #             eos_token_id=self.eos_token_id,
    #             **kwargs
    #         )
    #         # fmt: on

    #     generated_text = tokenizer.decode(generated_ids[0, input_ids.shape[1] :], skip_special_tokens=True).strip()

    #     return generated_text

    def allocate_inference_cache(self, *args, **kwargs):
        return self.llm_backbone.allocate_inference_cache(*args, **kwargs)
       
    @classmethod
    def from_pretrained(
        cls,
        pretrained_checkpoint: Path,
        model_id: str,
        llm_backbone_checkpoint: Path,
        speech_backbone: HFSpeechBackbone,
        llm_backbone: LLMBackbone,
        enable_mixed_precision_training: bool = True,
        arch_specifier: str = "gelu-mlp",
        llm_only: bool = False,
        # llm_cls: Optional[Type[PreTrainedModel]] = MambaForCausalLM,
    ) -> CobraSLM:
        """Initialize a CobraVLM from a pretrained checkpoint, freezing all weights, tailored for inference."""
        slm = cls(
            model_id,
            speech_backbone,
            llm_backbone,
            enable_mixed_precision_training=enable_mixed_precision_training,
            arch_specifier=arch_specifier,
            # arch_specifier="mamba",
        )

        # tmp = llm_cls.from_pretrained(llm_backbone_checkpoint)

        # Load from Checkpoint (Custom --> should load both *projector* and *llm* weights)
        model_state_dict = torch.load(pretrained_checkpoint, map_location="cpu")["model"]
        llm_state_dict = torch.load(os.path.join(llm_backbone_checkpoint, 'pytorch_model.bin'), map_location="cpu")
        # assert (
        #     "projector" in model_state_dict and "llm_backbone" in model_state_dict
        # ), "CobraVLM `from_pretrained` expects checkpoint with keys for `projector` AND `llm_backbone`!"

        adjusted_llm_state_dict = {}
        for key in llm_state_dict.keys():
            new_key = 'llm.' + key
            adjusted_llm_state_dict[new_key] = llm_state_dict[key]

        # Load the state dictionaries into the model
        if not llm_only:
            slm.projector.load_state_dict(model_state_dict["projector"], strict=False)
            slm.llm_backbone.load_state_dict(adjusted_llm_state_dict, strict=False)
            
        # slm.llm_backbone.load_state_dict(model_state_dict["llm_backbone"], False)

        # Freeze Weights
        slm.requires_grad_(False)
        slm.eval()

        return slm

    def get_prompt_builder(self, system_prompt: Optional[str] = None) -> PromptBuilder:
        prompt_initializer: Type[PromptBuilder] = self.llm_backbone.prompt_builder_fn
        return prompt_initializer(self.model_family, system_prompt=system_prompt)

    def freeze_backbones(self, stage: str) -> None:
        """
        This function sets `requires_grad_` on each of the component modules explicitly, depending on stage.

        We support two separate stages --> "align" and "finetune".
            => "align" --> vision_backbone*, llm_backbone* are frozen; only the `projector` is trained.
            => "finetune" --> vision_backbone* is frozen; both `projector` and `llm_backbone` are trained.

        :param stage: Pretraining stage in < "align" | "finetune" | "full-finetune" >
        """
        if stage == "align":
            self.speech_backbone.requires_grad_(False)
            self.llm_backbone.requires_grad_(False)
            self.projector.requires_grad_(True)

            # Add to `self.trainable_module_keys`
            self.trainable_module_keys = ["projector"]

            # Update Trackers
            self.speech_backbone_requires_grad = False

            # Explicitly Log Frozen / Trainable Components
            overwatch.info(f"[Frozen]    🥶 =>> Speech Backbone `{self.speech_backbone.identifier}`", ctx_level=1)
            overwatch.info(f"[Frozen]    🥶 =>> LLM Backbone `{self.llm_backbone.identifier}`", ctx_level=1)
            overwatch.info(f"[TRAINABLE] 🔥 =>> Projector `{self.arch_specifier}`", ctx_level=1)

        elif stage == "finetune":
            self.speech_backbone.requires_grad_(False)
            self.llm_backbone.requires_grad_(True)
            self.projector.requires_grad_(True)

            # Add to `self.trainable_module_keys`
            self.trainable_module_keys = ["projector", "llm_backbone"]

            # Update Trackers
            self.speech_backbone_requires_grad = False

            # Explicitly Log Frozen / Unfrozen Components
            overwatch.info(f"[Frozen]    🥶 =>> Speech Backbone `{self.speech_backbone.identifier}`", ctx_level=1)
            overwatch.info(f"[TRAINABLE] 🔥 =>> LLM Backbone `{self.llm_backbone.identifier}`", ctx_level=1)
            overwatch.info(f"[TRAINABLE] 🔥 =>> Projector `{self.arch_specifier}`", ctx_level=1)

        elif stage == "full-finetune":
            self.speech_backbone.dtype = torch.float32
            self.speech_backbone.requires_grad_(True)
            self.llm_backbone.requires_grad_(True)
            self.projector.requires_grad_(True)

            # Add to `self.trainable_module_keys`
            self.trainable_module_keys = ["speech_backbone", "projector", "llm_backbone"]

            # Update Trackers
            self.speech_backbone_requires_grad = True

            # Explicitly Log Frozen / Unfrozen Components
            overwatch.info(f"[TRAINABLE] 🔥 =>> Speech Backbone `{self.speech_backbone.identifier}`", ctx_level=1)
            overwatch.info(f"[TRAINABLE] 🔥 =>> LLM Backbone `{self.llm_backbone.identifier}`", ctx_level=1)
            overwatch.info(f"[TRAINABLE] 🔥 =>> Projector `{self.arch_specifier}`", ctx_level=1)

        elif stage == "arch-pretrain":
            self.speech_backbone.requires_grad_(False)
            self.llm_backbone.requires_grad_(False)
            self.projector.requires_grad_(True)

            # Add to `self.trainable_module_keys`
            self.trainable_module_keys = ["projector"]

            # Update Trackers
            self.speech_backbone_requires_grad = False

            # Explicitly Log Frozen / Trainable Components
            overwatch.info(f"[Frozen]    🥶 =>> Vision Backbone `{self.speech_backbone.identifier}`", ctx_level=1)
            overwatch.info(f"[Frozen]    🥶 =>> LLM Backbone `{self.llm_backbone.identifier}`", ctx_level=1)
            overwatch.info(f"[TRAINABLE] 🔥 =>> Projector `{self.arch_specifier}`", ctx_level=1)

        else:
            raise ValueError(f"Stage `{stage}` is not supported for LLaVa! Try < align | finetune >")

    def load_from_checkpoint(self, stage: str, run_dir: Path, pretrained_checkpoint: Optional[Path] = None) -> None:
        """Load weights from checkpoint (if required by the given stage)."""
        assert stage in {"align", "finetune", "full-finetune"}, f"Stage {stage} is not supported!"

        # If we're running a `no-align` architecture, we're good!
        if self.arch_specifier.startswith("no-align"):
            overwatch.info(
                f"CobraSLM with `{self.arch_specifier = }` does not require pretrained weights!", ctx_level=1
            )
            return

        # Otherwise, handle stage-specific logic!
        if stage == "align" or stage == "arch-pretrain":
            overwatch.info("Stage `align` does not require pretrained weights =>> Starting Training", ctx_level=1)
            return

        # Otherwise, load from `pretrained_checkpoint` or match on `run_dir` (s/+stage-finetune/+stage-align/g)
        overwatch.info("Stage `finetune` requires `align` pretrained weights", ctx_level=1)

        # Config specifies path to a checkpoint to load
        if pretrained_checkpoint is not None:
            overwatch.info(f"Loading from Provided Checkpoint `{pretrained_checkpoint}`", ctx_level=1)
            model_state_dict = torch.load(pretrained_checkpoint)["model"]
            self.projector.load_state_dict(model_state_dict["projector"], False)

            return

        # [Contract] If no `pretrained_checkpoint`, assume `align` lives in the run directory; string substitution!
        dataset, model, scale, _, seed = run_dir.name.split("+")
        align_dirs = [
            d
            for d in run_dir.parent.iterdir()
            if (d.name.startswith(f"{dataset}+{model}+{scale}") and d.name.endswith(f"+stage-align+{seed}"))
        ]
        assert len(align_dirs) == 1, "Multiple or No Valid Pretrained Directories Exist -- Double Check `runs`!"
        if (pretrained_checkpoint := (align_dirs[0] / "checkpoints" / "latest-checkpoint.pt")).exists():
            overwatch.info(f"Loading from Discovered Checkpoint `{pretrained_checkpoint}`", ctx_level=1)
            model_state_dict = torch.load(pretrained_checkpoint)["model"]
            self.projector.load_state_dict(model_state_dict["projector"], False)
        else:
            raise ValueError(f"Could not find valid `align` checkpoint at {pretrained_checkpoint}!")

    def get_fsdp_wrapping_policy(self) -> Callable:
        """Return an FSDP _or_policy over the policies returned by each individual backbone (and our VLM policy)."""
        speech_fsdp_wrapping_policy = self.speech_backbone.get_fsdp_wrapping_policy()
        llm_fsdp_wrapping_policy = self.llm_backbone.get_fsdp_wrapping_policy()

        # Get Cobra Wrapping Policy =>> just a module wrapping policy around `self.projector`
        cobra_fsdp_wrapping_policy = partial(
            _module_wrap_policy,
            module_classes={LinearProjector, MLPProjector, FusedMLPProjector, FusedLDPProjector, MambaProjectorV2},
        )

        # Return union (_or_) over constituent policies
        #   => Note: there is *not* a fall-through policy; any module that isn't covered by the above constituents will
        #            automatically be folded into the root VLM FSDP instance.
        return partial(
            _or_policy,
            policies=[
                speech_fsdp_wrapping_policy,
                llm_fsdp_wrapping_policy,
                cobra_fsdp_wrapping_policy,
            ],
        )
    
    def get_speech_projection_embeds(self, speech_embeddings: torch.Tensor):
        B, T, C = speech_embeddings.shape # [B, 1500, 1280]
        kernel = round(T * self.second_per_window / 30.0) # 17
        stride = round(T * self.second_stride / 30.0)   # 17 
        kernel = (1, kernel)
        stride = (1, stride)
        speech_embeddings_tr = speech_embeddings.transpose(1, 2).unsqueeze(2)
        speech_embeddings_unf = torch.nn.functional.unfold(speech_embeddings_tr, kernel_size=kernel, dilation=1, padding=0, stride=stride)

        _, _, L = speech_embeddings_unf.shape
        speech_embeddings_unf = speech_embeddings_unf.view(B, -1, kernel[1], L)
        speech_embeddings_unf = torch.permute(speech_embeddings_unf, [0, 3, 2, 1]) # [8, 88, 17, 1280]
        speech_embeddings = speech_embeddings_unf.reshape(-1, kernel[1], C) # [704, 17, 1280]
    

        # Projection Logic :: [bsz, num_patches, llm_embed_dim] =>> num_patches = (2 *) (256 + 1) for ViT-L + CLS
        projected_speech_embeddings = self.projector(speech_embeddings) # q-mamaba: query length = 1 # [8, 88, 2560] 

        projected_speech_embeddings = projected_speech_embeddings.view(B, -1, projected_speech_embeddings.shape[-1])
        return projected_speech_embeddings


    # Note =>> We're not explicitly subclassing `PreTrainedModel` because we don't need the bloat; however, `forward()`
    #          *must* match the signature of a `{Model}ForCausalLM` so that we can inherit from `GenerationMixin`

    # ruff: noqa: C901
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        input_values: Optional[Union[List[torch.FloatTensor], torch.FloatTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        multimodal_indices: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inference_params = None,
        num_last_tokens: int = 0,
        num_segments: Optional[torch.LongTensor] = None,
    ) -> CausalLMOutputWithPast:
        """Run a forward pass through the VLM, returning a CausalLMOutputWithPast instance (contains loss)."""
        # Handle Multimodal Indices is None --> pretend like the batch is fully multimodal (always image + text)!
        concating_input_values = None
        
        if multimodal_indices is None:
            multimodal_indices = torch.arange(len(input_ids), dtype=torch.long, device=input_ids.device)
        
        if num_segments is not None:
            num_segments = num_segments.to(input_ids.device)

        # Handle Multimodal Indices is Empty (len == 0) --> simple unimodal forward
        elif len(multimodal_indices) == 0:
            return self.llm_backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=None,
                past_key_values=past_key_values,
                inputs_embeds=None,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                inference_params=inference_params,
                num_last_tokens=num_last_tokens
            )
        # Run Speech Feature Extraction - # TODO: for the input over 30s, make sure get processed here correctly
        with torch.set_grad_enabled(self.speech_backbone_requires_grad):
            if isinstance(input_values, dict):
                patch_features = self.speech_backbone({k: input_values[k][multimodal_indices] for k in input_values})
            elif input_values is None:  # For cache phase in mamba's generate()
                return self.llm_backbone(
                input_ids=input_ids,
                attention_mask=None,
                position_ids=None,
                past_key_values=past_key_values,
                inputs_embeds=None,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                inference_params=inference_params,
                num_last_tokens=num_last_tokens,
            )
           
            elif isinstance(input_values, torch.Tensor):
                speech_embeddings = self.speech_backbone(input_values=input_values).last_hidden_state
                # projected_speech_embeddings = self.get_speech_projection_embeds(speech_embeddings)
            
            elif isinstance(input_values, list):
                concating_input_values = True
                input_values = torch.cat(input_values, dim=0)
                speech_embeddings = self.speech_backbone(input_values=input_values).last_hidden_state
        
                # =============================================================================================================
        if self.arch_specifier.endswith("qmamba"):
            projected_speech_embeddings = self.get_speech_projection_embeds(speech_embeddings)
            if concating_input_values:
                chunked_projected_speech_embeddings = torch.chunk(projected_speech_embeddings, chunks=input_ids.shape[0] , dim=0)
                projected_speech_embeddings = torch.stack([chunked_projected_speech_embedding.view(-1, projected_speech_embeddings.shape[-1]) for chunked_projected_speech_embedding in chunked_projected_speech_embeddings], dim=0)
                assert len(projected_speech_embeddings) == input_ids.shape[0] # batch size 
            else:
                pass



        # Get Input Embeddings from LLM Backbone :: [bsz, input_seq_len, llm_embed_dim]
        input_embeddings = self.llm_backbone.embed_input_ids(input_ids)
        speech_start_embedding, speech_end_embedding = self.get_speech_embeds()

        speech_start_embedding = speech_start_embedding.unsqueeze(0).expand(len(input_ids), -1, -1)
        speech_end_embedding = speech_end_embedding.unsqueeze(0).expand(len(input_ids), -1, -1)                
        # Build Multimodal Embeddings
        multimodal_embeddings = torch.cat(
            [
                speech_start_embedding,
                projected_speech_embeddings,
                speech_end_embedding, 
                input_embeddings[multimodal_indices, :, :],
            ],
            dim=1,
        ) # [8, 173, 2560]
        #   => We'll ignore the per-token outputs for each of the patch embeddings as well!
        multimodal_labels = None
        if labels is not None:
            projected_patch_labels = torch.full(
                (projected_speech_embeddings.shape[0], projected_speech_embeddings.shape[1]+ 2), # + 2 for speech tokens
                IGNORE_INDEX,
                dtype=labels.dtype,
                device=labels.device,
            )
            multimodal_labels = torch.cat(
                [projected_patch_labels, labels[multimodal_indices, :]], dim=1
            )
        # === Add Unimodal Handling ===
        # Create Fused Embeddings, Attention Mask, and Labels by Merging with "unimodal" Inputs (if applicable)
        unimodal_indices = torch.tensor(
            [idx for idx in range(len(input_ids)) if idx not in multimodal_indices],
            dtype=torch.long,
            device=multimodal_indices.device,
        )

        # No "unimodal" data --> Fused == Multimodal
        if len(unimodal_indices) == 0:
            fused_embeddings = multimodal_embeddings
            fused_labels = multimodal_labels

        else:
            # Otherwise --> Merge w/ unimodal data

            # This doesn't matter --> but in the "normal" case this is the embedding of the <PAD> token
            #   => NOTE :: Verified that `zeros/randn/empty/<PAD> embedding` all return the same result!
            unimodal_embeddings_pad = torch.zeros(
                (len(unimodal_indices), projected_speech_embeddings.shape[1], input_embeddings.shape[2]),
                dtype=input_embeddings.dtype,
                device=input_embeddings.device,
            )
            unimodal_labels_pad = torch.full(
                (len(unimodal_indices), projected_speech_embeddings.shape[1]),
                IGNORE_INDEX,
                dtype=labels.dtype,
                device=labels.device,
            )

            unimodal_embeddings = torch.cat([input_embeddings[unimodal_indices], unimodal_embeddings_pad], dim=1)
            unimodal_labels = torch.cat([labels[unimodal_indices], unimodal_labels_pad], dim=1)

            # Create "Fused" Tensors by Stacking Multimodal & Unimodal
            fused_embeddings = torch.vstack([multimodal_embeddings, unimodal_embeddings])
            fused_labels = torch.vstack([multimodal_labels, unimodal_labels])
        # Run LLM Forward --> returns CausalLMOutputWithPast!
        return self.llm_backbone(
            input_ids=None,
            attention_mask=None,
            position_ids=None,
            past_key_values=past_key_values,
            inputs_embeds=fused_embeddings,
            labels=fused_labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            inference_params=inference_params,
            num_last_tokens=num_last_tokens,
        )

    # === GenerationMixin Methods ===
    #   => Note: The following methods override the functionality of `transformers.GenerationMixin`; these expect the
    #            contract in each of the function signatures, and also expect our `forward` function to roughly take
    #            the same arguments as the underlying LLM (see `LlamaModelForCausalLM` as an example)

    def prepare_inputs_for_generation(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        input_values: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        **kwargs: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Borrowed from `LlamaForCausalLM` --> in general, just handles caching logic during generation."""
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        # Make sure `input_values` are preserved in `model_inputs`
        model_inputs.update(
            {
                "attention_mask": attention_mask,
                "input_values": input_values,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
            }
        )

        return model_inputs

    @torch.inference_mode()
    def generate_batch(
        self,
        input_values: Union[torch.Tensor, Dict[str, torch.Tensor]],
        texts: List[str],
        return_string_probabilities: Optional[List[str]] = None,
        **kwargs: str,
    ) -> Union[List[str], List[List[float]]]:
        # For now, only support generation with a batch size of 1 for simplicity
        tokenizer = self.llm_backbone.tokenizer

        # Prepare Inputs
        batch_input_ids = [
            tokenizer(text, truncation=True, return_tensors="pt").input_ids.to(self.device) for text in texts
        ]
        if isinstance(input_values, torch.Tensor):
            input_values = input_values[None, ...].to(self.device)
        elif isinstance(input_values, dict):
            input_values = {k: v[None, ...].to(self.device) for k, v in input_values.items()}
        elif isinstance(input_values, list):
            input_values = [v[None, ...].to(self.device) for v in input_values]
        else:
            raise ValueError(f"Unsupported `input_values` type = {type(input_values)}")

        # Create Output Lists
        gen_texts, gen_probabilities = [], []

        # Invoke super().generate --> taps into `GenerationMixin` which (redirects) to `forward()`
        autocast_dtype = self.llm_backbone.half_precision_dtype
        with torch.autocast("cuda", dtype=autocast_dtype, enabled=self.enable_mixed_precision_training):
            for idx, input_ids in enumerate(batch_input_ids):
                if isinstance(input_values, torch.Tensor):
                    input_values = input_values[idx]
                elif isinstance(input_values, dict):
                    input_values = {k: input_values[k][idx] for k in input_values}
                elif isinstance(input_values, list):
                    input_values = [input_value[idx] for input_value in input_values]
                else:
                    raise ValueError(f"Unsupported `input_values` type = {type(input_values)}")

                # Handle `return_string_probabilities`
                if return_string_probabilities is None:
                    full_out_ids = self.mamba_generate(input_ids=input_ids, input_values=input_values, eos_token_id=self.eos_token_id, **kwargs)
                    gen_ids = full_out_ids[0, input_ids.shape[1] :]
                    # Decode `gen_ids` and strip any <EOS> tokens
                    gen_texts.append(tokenizer.decode(gen_ids, skip_special_tokens=True).strip())

                else:
                    full_out_dict = self.mamba_generate(
                        input_ids=input_ids,
                        input_values=input_values,
                        output_scores=True,
                        return_dict_in_generate=True,
                        eos_token_id=self.eos_token_id,
                        **kwargs,
                    )

                    # Generation pattern should usually be [TOKEN] <EOS> for True/False and Yes/No Generations
                    gen_ids = full_out_dict.sequences[0, input_ids.shape[1] :]

                    # [Debug] Verify that the first token generated is in `self.string2idx.values()`
                    # assert gen_ids[0] in self.string2idx.values(), "Generated ID not in mapping!"

                    # Decode `gen_ids` and strip any <EOS> tokens
                    gen_texts.append(tokenizer.decode(gen_ids, skip_special_tokens=True).strip())

                    # Get all token probabilities --> softmax over logits
                    token_probs = torch.softmax(full_out_dict.scores[0][0], dim=0)

                    # Get *normalized* probabilities for all values in `return_token_probabilities`
                    slice_idxs = torch.tensor([self.string2idx[s] for s in return_string_probabilities])
                    string_probs_unnormalized = token_probs[slice_idxs]
                    string_probs = string_probs_unnormalized / string_probs_unnormalized.sum()
                    gen_probabilities.append(string_probs.cpu().numpy().tolist())

        return gen_texts if return_string_probabilities is None else gen_probabilities
    
    def generate_batch_no_text(
        self,
        input_values: Union[torch.Tensor, Dict[str, torch.Tensor]],
        texts: List[str],
        return_string_probabilities: Optional[List[str]] = None,
        **kwargs: str,
    ) -> Union[List[str], List[List[float]]]:
        # Prepare Inputs
        if isinstance(input_values, torch.Tensor):
            input_values = input_values[None, ...].to(self.device)
        elif isinstance(input_values, dict):
            input_values = {k: v[None, ...].to(self.device) for k, v in input_values.items()}
        else:
            raise ValueError(f"Unsupported `input_values` type = {type(input_values)}")

        # Create Output Lists
        gen_texts, gen_probabilities = [], []

        # Invoke super().generate --> taps into `GenerationMixin` which (redirects) to `forward()`
        autocast_dtype = self.llm_backbone.half_precision_dtype
        with torch.autocast("cuda", dtype=autocast_dtype, enabled=self.enable_mixed_precision_training):
            for idx in range(input_values.shape[0] if isinstance(input_values, torch.Tensor) else len(next(iter(input_values.values())))):
                if isinstance(input_values, torch.Tensor):
                    current_input_values = input_values[idx]
                elif isinstance(input_values, dict):
                    current_input_values = {k: input_values[k][idx] for k in input_values}
                else:
                    raise ValueError(f"Unsupported `input_values` type = {type(input_values)}")

                # Handle `return_string_probabilities`
                if return_string_probabilities is None:
                    full_out_ids = self.mamba_generate(
                        input_values=current_input_values, 
                        eos_token_id=self.eos_token_id, 
                        **kwargs
                    )
                    gen_ids = full_out_ids[0, 1:]  # Adjust slicing as necessary

                    # Decode `gen_ids` and strip any <EOS> tokens
                    gen_texts.append(self.llm_backbone.tokenizer.decode(gen_ids, skip_special_tokens=True).strip())

                else:
                    full_out_dict = self.mamba_generate(
                        input_values=current_input_values,
                        output_scores=True,
                        return_dict_in_generate=True,
                        eos_token_id=self.eos_token_id,
                        **kwargs,
                    )

                    # Generation pattern should usually be [TOKEN] <EOS> for True/False and Yes/No Generations
                    gen_ids = full_out_dict.sequences[0, 1:]  # Adjust slicing as necessary

                    # Decode `gen_ids` and strip any <EOS> tokens
                    gen_texts.append(self.llm_backbone.tokenizer.decode(gen_ids, skip_special_tokens=True).strip())

                    # Get all token probabilities --> softmax over logits
                    token_probs = torch.softmax(full_out_dict.scores[0][0], dim=0)

                    # Get *normalized* probabilities for all values in `return_token_probabilities`
                    slice_idxs = torch.tensor([self.string2idx[s] for s in return_string_probabilities])
                    string_probs_unnormalized = token_probs[slice_idxs]
                    string_probs = string_probs_unnormalized / string_probs_unnormalized.sum()
                    gen_probabilities.append(string_probs.cpu().numpy().tolist())

        return gen_texts if return_string_probabilities is None else gen_probabilities


    @torch.inference_mode()
    def generate(self, prompt_text: str, input_values=None, **kwargs: str) -> str:
        # For now, only support generation with a batch size of 1 for simplicity
        tokenizer = self.llm_backbone.tokenizer

        # Prepare Inputs
        input_ids = tokenizer(prompt_text, truncation=True, return_tensors="pt").input_ids.to(self.device)
        if isinstance(input_values, torch.Tensor):
            input_values = input_values[None, ...].to(self.device)
        elif isinstance(input_values, dict):
            input_values = {k: v[None, ...].to(self.device) for k, v in input_values.items()}
        else:
            raise ValueError(f"Unsupported `input_values` type = {type(input_values)}")

        # Invoke super().generate --> taps into `GenerationMixin` which (redirects) to `forward()`
        autocast_dtype = self.llm_backbone.half_precision_dtype
        input_values = input_values.to(autocast_dtype)
        with torch.autocast("cuda", dtype=autocast_dtype, enabled=self.enable_mixed_precision_training):
            # fmt: off
            generated_ids = self.mamba_generate(
                input_ids=input_ids,            # Shape: [1, seq]
                input_values=input_values,      # Shape: [1, 3, res, res] or Dict[str, Shape[1, 3, res, res]]
                eos_token_id=self.eos_token_id,
                **kwargs
            )
            # fmt: on

        generated_text = tokenizer.decode(generated_ids[0, input_ids.shape[1] :], skip_special_tokens=True).strip()

        return generated_text

    def get_speech_tokens(self):
        speech_start_token = self.llm_backbone.tokenizer("<Speech>", add_special_tokens=False)["input_ids"][0]
        speech_end_token = self.llm_backbone.tokenizer("</Speech>", add_special_tokens=False)["input_ids"][0]
        return speech_start_token, speech_end_token

    def get_speech_embeds(self):
        speech_start_token, speech_end_token = self.get_speech_tokens()
        speech_start_embed = self.llm_backbone.embed_input_ids(torch.tensor(speech_start_token).to(self.device))
        speech_end_embed = self.llm_backbone.embed_input_ids(torch.tensor(speech_end_token).to(self.device))
        return speech_start_embed, speech_end_embed
    