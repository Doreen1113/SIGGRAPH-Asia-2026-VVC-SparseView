#!/bin/bash
#SBATCH --job-name=train_env
#SBATCH --account=GOV114009
#SBATCH --partition=8gpus
#SBATCH --exclude=25a-hgpn144,25a-hgpn111
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=/work/doreen071/vvc/jobs/spreadenv_%j.log
# Like train_spread, but spec tokens of the form FTGSPP_X=Y are exported as env vars instead of hydra overrides.
# Usage: sbatch --nodes=N --ntasks=N --export=ALL,SPEC=<file> train_spread_env.sbatch
srun --kill-on-bad-exit=0 --wait=0 bash -c '
module load cuda/12.6 2>/dev/null
export PATH=/home/doreen071/.conda/envs/vvc312/bin:$PATH
export TORCH_CUDA_ARCH_LIST=9.0
export TORCH_EXTENSIONS_DIR=$(mktemp -d /tmp/tex.XXXXXX)/py312_cu121
mkdir -p $TORCH_EXTENSIONS_DIR && cp -r /work/doreen071/vvc/envcache/gsplat_prebuilt/gsplat_cuda $TORCH_EXTENSIONS_DIR/
export HYDRA_FULL_ERROR=1 OMP_NUM_THREADS=10
V=/work/doreen071/vvc
LINE=$(sed -n "$((SLURM_PROCID+1))p" "$SPEC")
[ -z "$LINE" ] && { echo "no spec line for rank $SLURM_PROCID"; exit 0; }
set -- $LINE
M=$1; C=$2; shift 2
OVR=""
for tok in "$@"; do
  case "$tok" in
    FTGSPP_*=*|MINPLY=*) export "$tok"; echo "env: $tok" ;;
    *) OVR="$OVR $tok" ;;
  esac
done
R=$V/runs/siga_$C; OUT=$R/run_$M
if [ -f $OUT/gaussians.pt ]; then echo "skip $M/$C (done)"; exit 0; fi
echo "rank $SLURM_PROCID on $(hostname): $M $C [$OVR]"
PTS=${FTGSPP_POINTS_DIR:-$R/points_stride10}   # spec line may set FTGSPP_POINTS_DIR (e.g. points_stride5)
MINPLY=${MINPLY:-40}
until [ -d $PTS ] && [ $(ls $PTS/*.ply 2>/dev/null | wc -l) -ge $MINPLY ] && [ -z "$(find $PTS -mmin -2 -name "*.ply")" ]; do sleep 60; done
if [ -n "${FTGSPP_PSEUDO_DIR:-}" ]; then until [ -f $FTGSPP_PSEUDO_DIR/index.json ]; do sleep 60; done; fi
export FTGSPP_RUNS_ROOT=$V/runs
export FTGSPP_LPIPS_W=${FTGSPP_LPIPS_W:-0.3} FTGSPP_DEPTH_W=${FTGSPP_DEPTH_W:-0.2} FTGSPP_REG_OPACITY=${FTGSPP_REG_OPACITY:-0.05} FTGSPP_DENSE_W=${FTGSPP_DENSE_W:-0.1}
export FTGSPP_TRAIN_EVAL_INTERVAL=0 FTGSPP_CHECKPOINT_INTERVAL=0
export FTGSPP_DENSE_DEPTH_DIR=$V/depth_vggt/$C FTGSPP_POINTS_DIR=$PTS
cd $V/repo/baseline_code
python -m ftgspp.run data=siga_$C run.gpu=0 \
  data.root=$V/data/$C data.video_path=$V/data/$C data.calibration_path=$V/data/$C data.extracted_path=$V/data/$C/images \
  train.color_correction=true train.lpips_loss=true train.num_workers=16 train.prefetch_factor=4 $OVR run.output_path=$OUT > $V/jobs/train_${M}_${C}.log 2>&1
[ -f $OUT/gaussians.pt ] && echo "TRAIN_${M}_${C}_DONE" || echo "TRAIN_${M}_${C}_FAILED"
'
echo SPREAD_ENV_DONE
