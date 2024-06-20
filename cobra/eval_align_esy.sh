
torchrun --standalone --nnodes 1 --nproc-per-node 1 speech_eval_long_esy.py \
  --model_family "mamba" \
  --model_id "test_long" \
  --model_dir "/mnt/hdd/hsyoon/workspace/samsung/cobra/runs/qmamba+align-query2" \
  --speech_backbone_id "whisper-large-v2" \
  --speech_backbone_path "/mnt/hdd/hsyoon/workspace/models/whisper-large-v2" \
  --llm_backbone_id "mamba-2.8b-zephyr" \
  --llm_backbone_path "/mnt/hdd/hsyoon/workspace/models/mamba-2.8b-zephyr" \
  --device_batch_size 1 \
  --num_workers 0 \
  --results_dir "/mnt/hdd/hsyoon/workspace/samsung/cobra/runs/qmamba+align-query2/eval" \
  --root_dir "/mnt/hdd/hsyoon/workspace/ES/speech/datasets/" \
  --hf_token "/mnt/hdd/hsyoon/workspace/samsung/cobra/.hf_token" \