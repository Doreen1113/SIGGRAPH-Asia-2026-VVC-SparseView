#!/bin/bash
#SBATCH --job-name=points5
#SBATCH --account=GOV114009
#SBATCH --partition=8gpus
#SBATCH --exclude=25a-hgpn144,25a-hgpn111
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=/work/doreen071/vvc/jobs/points5_%j.log
# keyframe_stride has been 10 for every model we have ever trained: the Gaussian init only gets keyframe points
# every 10 frames, and init.duration is tied to it. These are volumetric VIDEOS of a MOVING person, so that
# temporal coarseness should hurt exactly where we are weak (FG) and not where we are strong (static background).
# Point generation costs only ~6.5 min/scene, so stride 5 is nearly free to try.
srun --kill-on-bad-exit=0 --wait=0 bash -c '
module load cuda/12.6 2>/dev/null
export PATH=/home/doreen071/.conda/envs/vvc312/bin:$PATH TORCH_CUDA_ARCH_LIST=9.0 HYDRA_FULL_ERROR=1 OMP_NUM_THREADS=10
export TORCH_EXTENSIONS_DIR=$(mktemp -d /tmp/tex.XXXXXX)/py312_cu121
mkdir -p $TORCH_EXTENSIONS_DIR && cp -r /work/doreen071/vvc/envcache/gsplat_prebuilt/gsplat_cuda $TORCH_EXTENSIONS_DIR/
V=/work/doreen071/vvc
C=$(sed -n "$((SLURM_PROCID+1))p" /work/doreen071/vvc/work/slurm/specs/scenes6.txt)
[ -z "$C" ] && exit 0
R=$V/runs/siga_$C
export FTGSPP_RUNS_ROOT=$V/runs
cd $V/repo/baseline_code
python -m ftgspp.run data=siga_$C run.gpu=0 pipeline.stages=[points] \
  data.root=$V/data/$C data.video_path=$V/data/$C data.calibration_path=$V/data/$C data.extracted_path=$V/data/$C/images \
  init.keyframe_stride=5 init.points_path=$R/points_stride5 \
  run.output_path=$R/run_POINTS5 > $V/jobs/points5_$C.log 2>&1
n=$(ls $R/points_stride5/*.ply 2>/dev/null | wc -l)
echo "POINTS5 $C: $n plys"
'
echo POINTS5_DONE $(date +%H:%M)
