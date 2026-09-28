#!/bin/bash
#SBATCH --job-name=points3
#SBATCH --account=GOV114009
#SBATCH --partition=8gpus
#SBATCH --exclude=25a-hgpn144,25a-hgpn111
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=/work/doreen071/vvc/jobs/points3_%j.log
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
  init.keyframe_stride=3 init.points_path=$R/points_stride3 \
  run.output_path=$R/run_POINTS3 > $V/jobs/points3_$C.log 2>&1
n=$(ls $R/points_stride3/*.ply 2>/dev/null | wc -l)
echo "POINTS3 $C: $n plys"
'
echo POINTS3_DONE $(date +%H:%M)
