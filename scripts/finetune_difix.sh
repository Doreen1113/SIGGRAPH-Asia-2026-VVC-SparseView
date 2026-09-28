#!/bin/bash
# Fine-tune Difix on OUR OWN render->GT pairs. Organizers explicitly permit using this track's
# training views to fine-tune a generative prior. A generic "remove degradation" model trades SSIM for
# LPIPS by hallucinating plausible-but-misaligned detail; a fine-tuned one has seen the real targets for
# this rig/subject, so it should improve LPIPS at a lower SSIM cost — i.e. move the tradeoff curve.
cd /home/intern_2603055/projects/Difix3D
export HF_HOME=/home/intern_2603055/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline
while [ $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) -gt 26000 ]; do sleep 180; done
/home/intern_2603055/vvc/envs/difix/bin/python src/train_difix.py \
  --dataset_path /home/intern_2603055/vvc/difix_ft/dataset.json \
  --output_dir /home/intern_2603055/vvc/difix_ft/ckpt \
  --tracker_run_name vvc_ft \
  --pretrained_model_name_or_path "$(ls -d /home/intern_2603055/.cache/huggingface/hub/models--nvidia--difix_ref/snapshots/*/ | head -1)" \
  --mv_unet --timestep 199 --resolution 320 --train_batch_size 1 \
  --max_train_steps 1500 --num_training_epochs 32 --checkpointing_steps 250 --learning_rate 1e-5 --eval_freq 500 --viz_freq 500 --resume /home/intern_2603055/vvc/difix_ft/ckpt/checkpoints \
  --lambda_gram 0 --mixed_precision bf16 --gradient_accumulation_steps 8 --num_samples_eval 8 \
  --enable_xformers_memory_efficient_attention --gradient_checkpointing \
  2>&1 | tail -40
echo "DIFIX FINETUNE DONE $(date -u +%H:%M)"
