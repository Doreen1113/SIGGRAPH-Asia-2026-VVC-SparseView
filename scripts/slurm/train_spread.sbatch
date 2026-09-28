#!/bin/bash
#SBATCH --job-name=train_spread
#SBATCH --account=GOV114009
#SBATCH --partition=dev
#SBATCH --exclude=25a-hgpn144,25a-hgpn111
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --time=03:55:00
#SBATCH --output=/work/doreen071/vvc/jobs/spread_%j.log
# One task per NODE (full CPU each => full ~6.7 it/s speed), one training run per task.
# Usage: sbatch --nodes=<N> --export=ALL,SPEC=<file> train_spread.sbatch   (SPEC has N lines: "MEMBER CASE OVERRIDES...")
srun --kill-on-bad-exit=0 --wait=0 bash -c '
module load cuda/12.6 2>/dev/null
export PATH=/home/doreen071/.conda/envs/vvc312/bin:$PATH
export TORCH_CUDA_ARCH_LIST=9.0
export TORCH_EXTENSIONS_DIR=$(mktemp -d /tmp/tex.XXXXXX)/py312_cu121
mkdir -p $TORCH_EXTENSIONS_DIR && cp -r /work/doreen071/vvc/envcache/gsplat_prebuilt/gsplat_cuda $TORCH_EXTENSIONS_DIR/
export HYDRA_FULL_ERROR=1 OMP_NUM_THREADS=10
V=/work/doreen071/vvc
read -r M C OVR < <(sed -n "$((SLURM_PROCID+1))p" "$SPEC")
[ -z "$M" ] && { echo "no spec line for rank $SLURM_PROCID"; exit 0; }
R=$V/runs/siga_$C; OUT=$R/run_$M
if [ -f $OUT/gaussians.pt ]; then echo "skip $M/$C (done)"; exit 0; fi
echo "rank $SLURM_PROCID on $(hostname): $M $C [$OVR]"
until [ -d $R/points_stride10 ] && [ $(ls $R/points_stride10/*.ply 2>/dev/null | wc -l) -ge 40 ] && [ -z "$(find $R/points_stride10 -mmin -2 -name "*.ply")" ]; do sleep 60; done
export FTGSPP_RUNS_ROOT=$V/runs FTGSPP_LPIPS_W=0.3 FTGSPP_DEPTH_W=0.2 FTGSPP_REG_OPACITY=0.05 FTGSPP_DENSE_W=0.1
export FTGSPP_TRAIN_EVAL_INTERVAL=0 FTGSPP_CHECKPOINT_INTERVAL=0
export FTGSPP_DENSE_DEPTH_DIR=$V/depth_vggt/$C FTGSPP_POINTS_DIR=$R/points_stride10
cd $V/repo/baseline_code
python -m ftgspp.run data=siga_$C run.gpu=0 \
  data.root=$V/data/$C data.video_path=$V/data/$C data.calibration_path=$V/data/$C data.extracted_path=$V/data/$C/images \
  train.color_correction=true train.lpips_loss=true train.num_workers=16 train.prefetch_factor=4 $OVR run.output_path=$OUT > $V/jobs/train_${M}_${C}.log 2>&1
[ -f $OUT/gaussians.pt ] && echo "TRAIN_${M}_${C}_DONE" || echo "TRAIN_${M}_${C}_FAILED"
'
echo SPREAD_DONE
