torchrun --standalone --nnodes 1 --nproc-per-node 8 scripts/pretrain.py \
  --model.vision_backbone_id "siglip-vit-so400m-384px" \
  --model.image_resize_strategy "resize-naive" \
  --model.llm_backbone_id "mamba-2.8b-zephyr" \
  --model.type "cobra+3b" \
  --model.finetune_global_batch_size 128 \
  --model.finetune_per_device_batch_size 8 \
  --dataset.type "llava-v15" \
  --model.align_learning_rate 0.0001 \
  --model.arch_specifier "qmamba" \
  --model.finetune_epochs 1 \
  --stage "finetune" \
  --seed 7 \
  --run_id "simple-qmamba+finetune+local" \
  --pretrained_checkpoint "runs/simple-qmamba+align+local/checkpoints/latest-checkpoint.pt" \
  
