torchrun --standalone --nnodes 1 --nproc-per-node 1 scripts/pretrain.py \
  --model.vision_backbone_id "dinosiglip-vit-so-384px" \
  --model.image_resize_strategy "resize-naive" \
  --model.llm_backbone_id "mamba-2.8b-zephyr" \
  --model.type "cobra+3b" \
  --model.finetune_global_batch_size 2 \
  --model.finetune_per_device_batch_size 2 \
  --dataset.type "llava-lvis4v-lrv"\
  --stage "finetune" \
  --run_id "cobra-finetune-debug" \
  --wandb_project "cobra-debug" \