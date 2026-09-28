#!/bin/bash
#SBATCH --job-name=dfx_distill
#SBATCH --account=GOV114009
#SBATCH --partition=dev
#SBATCH --exclude=25a-hgpn144,25a-hgpn111
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --time=03:55:00
#SBATCH --output=/work/doreen071/vvc/jobs/dfx_distill_%j.log
# Difix3D's 3D step: distil the Difix-TTA corrections at the test poses into a single 3D model.
# One task per node; each line of SPEC is "MEMBER CASE".
srun --kill-on-bad-exit=0 --wait=0 bash -c '
module load cuda/12.6 2>/dev/null
export PATH=/home/doreen071/.conda/envs/vvc312/bin:$PATH TORCH_CUDA_ARCH_LIST=9.0 HYDRA_FULL_ERROR=1 OMP_NUM_THREADS=10
export TORCH_EXTENSIONS_DIR=$(mktemp -d /tmp/tex.XXXXXX)/py312_cu121
mkdir -p $TORCH_EXTENSIONS_DIR && cp -r /work/doreen071/vvc/envcache/gsplat_prebuilt/gsplat_cuda $TORCH_EXTENSIONS_DIR/
V=/work/doreen071/vvc
read -r M C < <(sed -n "$((SLURM_PROCID+1))p" "$SPEC")
[ -z "$M" ] && exit 0
R=$V/runs/siga_$C; OUT=$R/run_$M; PD=$V/pseudo/$C
[ -f $OUT/gaussians.pt ] && { echo "skip $M/$C"; exit 0; }
if [ ! -f $PD/index.json ]; then
  python $V/work/tta_to_pseudo.py $C $V/submissions/NINE_TTA/renders $PD --every 2 --scale 0.5
fi
echo "rank $SLURM_PROCID on $(hostname): $M $C  pseudo=$(python -c "import json;print(len(json.load(open(\"$PD/index.json\"))))")"
export FTGSPP_RUNS_ROOT=$V/runs
export FTGSPP_LPIPS_W=0.3 FTGSPP_DEPTH_W=0.2 FTGSPP_REG_OPACITY=0.05 FTGSPP_DENSE_W=0.1
export FTGSPP_TRAIN_EVAL_INTERVAL=0 FTGSPP_CHECKPOINT_INTERVAL=0
export FTGSPP_DENSE_DEPTH_DIR=$V/depth_vggt/$C FTGSPP_POINTS_DIR=$R/points_stride10
export FTGSPP_PSEUDO_DIR=$PD FTGSPP_PSEUDO_W=0.5 FTGSPP_PSEUDO_LPIPS=0.1 FTGSPP_PSEUDO_START=3000
export FTGSPP_PSEUDO_MASK=0 FTGSPP_PSEUDO_COVERAGE=0 FTGSPP_PSEUDO_AGREE=0
cd $V/repo/baseline_code
python -m ftgspp.run data=siga_$C run.gpu=0 \
  data.root=$V/data/$C data.video_path=$V/data/$C data.calibration_path=$V/data/$C data.extracted_path=$V/data/$C/images \
  train.color_correction=true train.lpips_loss=true train.num_workers=16 train.prefetch_factor=4 data.scale=1.0 \
  run.output_path=$OUT > $V/jobs/train_${M}_${C}.log 2>&1
[ -f $OUT/gaussians.pt ] && echo "TRAIN_${M}_${C}_DONE" || echo "TRAIN_${M}_${C}_FAILED"
'
echo DFX_DISTILL_DONE
