torchrun --standalone --nnodes 1 --nproc_per_node 8 scripts/pretrain_speech_ver2.py \
  --model.speech_backbone_id "whisper-large-v2" \
  --model.speech_backbone_path "/data2/data_account/workspace/models/whisper-large-v2" \
  --model.llm_backbone_id "mamba-2.8b-zephyr" \
  --model.llm_backbone_path "/data2/data_account/workspace/models/mamba-2.8b-zephyr" \
  --model.type "cobra+3b" \
  --model.finetune_global_batch_size 128 \
  --model.finetune_per_device_batch_size 8 \
  --model.align_global_batch_size 256 \
  --model.align_per_device_batch_size 16 \
  --dataset.type "librispeech+gigaspeech" \
  --model.arch_specifier "qmamba" \
  --model.finetune_epochs 1 \
  --model.align_epochs 2 \
  --stage "align" \
  --seed 7 \
  --run_id "qmamba+align-libri+giga" \
  # --pretrained_checkpoint "runs/vlmamba+align+ln/checkpoints/latest-checkpoint.pt" \

# justMamba: just middle layer to mamba, no other difference
# --run_id "vlmamba-conv+align-debug" \