#!/bin/bash
#SBATCH --job-name=vvc_traintest
#SBATCH --account=GOV114009
#SBATCH --partition=dev
#SBATCH --exclude=25a-hgpn144,25a-hgpn111
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/work/doreen071/vvc/jobs/traintest_%j.log
module load cuda/12.6
export PATH=/home/doreen071/.conda/envs/vvc312/bin:$PATH
PY=/home/doreen071/.conda/envs/vvc312/bin/python
export TORCH_CUDA_ARCH_LIST=9.0
export FTGSPP_RUNS_ROOT=/work/doreen071/vvc/runs
export FTGSPP_LPIPS_W=0.3 FTGSPP_DEPTH_W=0.2 FTGSPP_REG_OPACITY=0.05 FTGSPP_DENSE_W=0.1
export FTGSPP_DENSE_DEPTH_DIR=/work/doreen071/vvc/depth_vggt/001_1_seq0
export FTGSPP_POINTS_DIR=/work/doreen071/vvc/runs/siga_001_1_seq0/points_stride10
export FTGSPP_TRAIN_EVAL_INTERVAL=0 FTGSPP_CHECKPOINT_INTERVAL=0
hostname; nvidia-smi --query-gpu=name,compute_mode --format=csv,noheader; export HYDRA_FULL_ERROR=1
cd /work/doreen071/vvc/repo/baseline_code
$PY -m ftgspp.run data=siga_001_1_seq0 run.gpu=0 data.scale=1.0 \
  data.root=/work/doreen071/vvc/data/001_1_seq0 \
  data.video_path=/work/doreen071/vvc/data/001_1_seq0 \
  data.calibration_path=/work/doreen071/vvc/data/001_1_seq0 \
  data.extracted_path=/work/doreen071/vvc/data/001_1_seq0/images \
  train.color_correction=true train.lpips_loss=true \
  run.output_path=/work/doreen071/vvc/runs/siga_001_1_seq0/run_H200_TEST
echo TRAINTEST_DONE
