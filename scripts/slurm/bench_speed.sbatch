#!/bin/bash
#SBATCH --job-name=bench
#SBATCH --account=GOV114009
#SBATCH --partition=dev
#SBATCH --exclude=25a-hgpn144,25a-hgpn111
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --gres=gpu:1
#SBATCH --time=00:25:00
#SBATCH --output=/work/doreen071/vvc/jobs/bench_%j.log
module load cuda/12.6 2>/dev/null
export PATH=/home/doreen071/.conda/envs/vvc312/bin:$PATH TORCH_CUDA_ARCH_LIST=9.0 HYDRA_FULL_ERROR=1
export TORCH_EXTENSIONS_DIR=$(mktemp -d /tmp/tex.XXXXXX)/py312_cu121
mkdir -p $TORCH_EXTENSIONS_DIR && cp -r /work/doreen071/vvc/envcache/gsplat_prebuilt/gsplat_cuda $TORCH_EXTENSIONS_DIR/
V=/work/doreen071/vvc; C=001_1_seq0; R=$V/runs/siga_$C
export FTGSPP_RUNS_ROOT=$V/runs FTGSPP_LPIPS_W=0.3 FTGSPP_DEPTH_W=0.2 FTGSPP_REG_OPACITY=0.05 FTGSPP_DENSE_W=0.1
export FTGSPP_TRAIN_EVAL_INTERVAL=0 FTGSPP_CHECKPOINT_INTERVAL=0
export FTGSPP_DENSE_DEPTH_DIR=$V/depth_vggt/$C FTGSPP_POINTS_DIR=$R/points_stride10
export OMP_NUM_THREADS=16
hostname; nproc; nvidia-smi --query-gpu=name --format=csv,noheader
cd $V/repo/baseline_code
# exclusive node + 16 dataloader workers, 1500 steps just to measure it/s
timeout 900 python -m ftgspp.run data=siga_$C run.gpu=0 data.scale=1.0 \
  data.root=$V/data/$C data.video_path=$V/data/$C data.calibration_path=$V/data/$C data.extracted_path=$V/data/$C/images \
  train.color_correction=true train.lpips_loss=true train.num_workers=16 train.prefetch_factor=4 train.iterations=1500 \
  run.output_path=$R/run_BENCH_FAST 2>&1 | tail -5
echo BENCH_DONE
