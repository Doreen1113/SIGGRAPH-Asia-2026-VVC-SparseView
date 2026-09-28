#!/bin/bash
#SBATCH --job-name=train_pack
#SBATCH --account=GOV114009
#SBATCH --partition=dev
#SBATCH --exclude=25a-hgpn144,25a-hgpn111
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --time=03:55:00
#SBATCH --output=/work/doreen071/vvc/jobs/train_pack_%j.log
# Usage: sbatch --export=ALL,SPEC=<file> train_pack.sbatch ; SPEC lines: "MEMBER CASE OVERRIDES..." (max 8 lines)
module load cuda/12.6 2>/dev/null
export PATH=/home/doreen071/.conda/envs/vvc312/bin:$PATH
export TORCH_CUDA_ARCH_LIST=9.0
export TORCH_EXTENSIONS_DIR=$(mktemp -d /tmp/tex.XXXXXX)/py312_cu121
mkdir -p $TORCH_EXTENSIONS_DIR && cp -r /work/doreen071/vvc/envcache/gsplat_prebuilt/gsplat_cuda $TORCH_EXTENSIONS_DIR/
export HYDRA_FULL_ERROR=1 OMP_NUM_THREADS=8
V=/work/doreen071/vvc
export FTGSPP_RUNS_ROOT=$V/runs FTGSPP_LPIPS_W=0.3 FTGSPP_DEPTH_W=0.2 FTGSPP_REG_OPACITY=0.05 FTGSPP_DENSE_W=0.1
export FTGSPP_TRAIN_EVAL_INTERVAL=0 FTGSPP_CHECKPOINT_INTERVAL=0
hostname; nvidia-smi --query-gpu=name,compute_mode --format=csv,noheader | head -1
i=0
while read -r M C OVR; do
  [ -z "$M" ] && continue
export  R=$V/runs/siga_$C; OUT=$R/run_$M
  if [ -f $OUT/gaussians.pt ]; then echo "skip $M/$C"; continue; fi
  (
    until [ -d $R/points_stride10 ] && [ $(ls $R/points_stride10/*.ply 2>/dev/null | wc -l) -ge 40 ] && [ -z "$(find $R/points_stride10 -mmin -2 -name '*.ply')" ]; do sleep 60; done
    cd $V/repo/baseline_code
export    CUDA_VISIBLE_DEVICES=$i FTGSPP_DENSE_DEPTH_DIR=$V/depth_vggt/$C FTGSPP_POINTS_DIR=$R/points_stride10 \
    python -m ftgspp.run data=siga_$C run.gpu=0 \
      data.root=$V/data/$C data.video_path=$V/data/$C data.calibration_path=$V/data/$C data.extracted_path=$V/data/$C/images \
      train.color_correction=true train.lpips_loss=true train.num_workers=16 train.prefetch_factor=4 $OVR run.output_path=$OUT > $V/jobs/train_${M}_${C}.log 2>&1
    [ -f $OUT/gaussians.pt ] && echo "TRAIN_${M}_${C}_DONE" || echo "TRAIN_${M}_${C}_FAILED"
  ) &
  i=$((i+1))
done < "$SPEC"
wait
echo PACK_DONE
