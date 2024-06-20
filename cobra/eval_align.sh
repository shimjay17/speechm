source ~/anaconda3/etc/profile.d/conda.sh
conda activate speech_mamba5

NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=1,2,3 torchrun --standalone --nnodes 1 --nproc-per-node 3 speech_eval2.py \
  --model_family "mamba" \
  --model_id "test" \
  --model_dir "/mnt/hdd/hsyoon/workspace/samsung/cobra/runs/qmamba+align" \
  --speech_backbone_id "whisper-large-v2" \
  --speech_backbone_path "/mnt/hdd/hsyoon/workspace/models/whisper-large-v2" \
  --llm_backbone_id "mamba-2.8b-zephyr" \
  --llm_backbone_path "/mnt/hdd/hsyoon/workspace/models/mamba-2.8b-zephyr" \
  --device_batch_size 1 \
  --num_workers 1 \
  --results_dir "/mnt/hdd/hsyoon/workspace/samsung/cobra/" \
  --root_dir "/mnt/hdd/hsyoon/workspace/ES/speech/datasets/" \
  --hf_token "/mnt/hdd/hsyoon/workspace/samsung/cobra/.hf_token" \