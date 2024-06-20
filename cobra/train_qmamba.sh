NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --standalone --nnodes 1 --nproc-per-node 4 scripts/pretrain.py \
  --model.vision_backbone_id "dinosiglip-vit-so-384px" \
  --model.image_resize_strategy "resize-naive" \
  --model.llm_backbone_id "mamba-2.8b-zephyr" \
  --model.type "cobra+3b" \
  --model.finetune_global_batch_size 128 \
  --model.finetune_per_device_batch_size 8 \
  --dataset.type "llava-v15" \
  --model.arch_specifier "qmamba" \
  --stage "finetune" \
  --seed 7 \
  --run_id "none+vim+scr+384+finetune+llava-v15" \
  --pretrained_checkpoint "runs/none+vim+scr+384+align/checkpoints/latest-checkpoint.pt" \

# if mlp channel mixer, mlp else none
# if vision mamba, vim else none
# if from pretrained ptr else scr
# if length 384...
