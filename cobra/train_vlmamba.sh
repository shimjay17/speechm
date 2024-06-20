torchrun --standalone --nnodes 1 --nproc-per-node 8 scripts/pretrain.py \
  --model.vision_backbone_id "dinosiglip-vit-so-384px" \
  --model.image_resize_strategy "resize-naive" \
  --model.llm_backbone_id "mamba-2.8b-zephyr" \
  --model.type "cobra+3b" \
  --model.finetune_global_batch_size 128 \
  --model.finetune_per_device_batch_size 8 \
  --dataset.type "llava-v15" \
  --model.arch_specifier "vlmamba" \
  --model.finetune_epochs 1 \
  --stage "finetune" \
  --seed 7 \
  --run_id "vlmamba+finetune+ln" \
  --pretrained_checkpoint "runs/vlmamba+align+ln/checkpoints/latest-checkpoint.pt" \

# justMamba: just middle layer to mamba, no other difference