#!/bin/bash
#SBATCH --job-name=vvc_train
#SBATCH --account=GOV114009
#SBATCH --partition=dev
#SBATCH --exclude=25a-hgpn144,25a-hgpn111
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --time=03:55:00
#SBATCH --output=/work/doreen071/vvc/jobs/train_%x_%j.log
# Usage: sbatch --job-name=<MEMBER>_<CASE> --export=ALL,MEMBER=G3M_s0,CASE=004_1_seq0,OVR="init.num_gaussians=3000000" train_member.sbatch
module load cuda/12.6 2>/dev/null
export PATH=/home/doreen071/.conda/envs/vvc312/bin:$PATH
export TORCH_CUDA_ARCH_LIST=9.0
export TORCH_EXTENSIONS_DIR=$(mktemp -d /tmp/tex.XXXXXX)/py312_cu121
mkdir -p $TORCH_EXTENSIONS_DIR && cp -r /work/doreen071/vvc/envcache/gsplat_prebuilt/gsplat_cuda $TORCH_EXTENSIONS_DIR/
export HYDRA_FULL_ERROR=1
V=/work/doreen071/vvc; R=$V/runs/siga_$CASE; OUT=$R/run_$MEMBER
export FTGSPP_RUNS_ROOT=$V/runs
export FTGSPP_LPIPS_W=0.3 FTGSPP_DEPTH_W=0.2 FTGSPP_REG_OPACITY=0.05 FTGSPP_DENSE_W=0.1
export FTGSPP_DENSE_DEPTH_DIR=$V/depth_vggt/$CASE FTGSPP_POINTS_DIR=$R/points_stride10
export FTGSPP_TRAIN_EVAL_INTERVAL=0 FTGSPP_CHECKPOINT_INTERVAL=0
hostname; nvidia-smi --query-gpu=name --format=csv,noheader; echo "MEMBER=$MEMBER CASE=$CASE OVR=$OVR"
# wait for keyframe points if another job is generating them (POINTS_OWNER=1 generates)
if [ "${POINTS_OWNER:-0}" != "1" ]; then
  until [ -d $R/points_stride10 ] && [ $(ls $R/points_stride10/*.ply 2>/dev/null | wc -l) -ge 40 ] && [ -z "$(find $R/points_stride10 -mmin -2 -name '*.ply')" ]; do sleep 60; done
fi
cd $V/repo/baseline_code
python -m ftgspp.run data=siga_$CASE run.gpu=0 \
  data.root=$V/data/$CASE data.video_path=$V/data/$CASE data.calibration_path=$V/data/$CASE data.extracted_path=$V/data/$CASE/images \
  train.color_correction=true train.lpips_loss=true train.num_workers=16 train.prefetch_factor=4 $OVR run.output_path=$OUT
ls -la $OUT/gaussians.pt && echo TRAIN_${MEMBER}_${CASE}_DONE
