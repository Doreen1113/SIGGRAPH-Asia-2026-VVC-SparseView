#!/bin/bash
#SBATCH --job-name=render_val
#SBATCH --account=GOV114009
#SBATCH --partition=dev
#SBATCH --exclude=25a-hgpn144,25a-hgpn111
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --output=/work/doreen071/vvc/jobs/render_val_%j.log
# Usage: sbatch --export=ALL,SCENE=001_1_seq0,NAME=ens4,MAIN=run_FULLRES_s0,EXTRA=run_J_s0,run_GATE_s0,run_G4M_s0 render_val.sbatch
# Dumps 24 val triplets (render of the ensemble, gt, ref) to fixtest_h200/<SCENE>/<NAME>, then FULL+FG metrics (alex+vgg).
module load cuda/12.6 2>/dev/null
export PATH=/home/doreen071/.conda/envs/vvc312/bin:$PATH TORCH_CUDA_ARCH_LIST=9.0 OMP_NUM_THREADS=8
V=/work/doreen071/vvc; R=$V/runs/siga_$SCENE; OUT=$V/fixtest_h200/$SCENE/$NAME
EX=$(echo $EXTRA | sed "s|run_|$R/run_|g")
python $V/work/dump_val_renders_h200.py $SCENE $R/$MAIN "$EX" $OUT --every 200
/home/doreen071/.conda/envs/vvc/bin/python $V/work/metrics_both.py $OUT render 2>&1 | grep -vE "Warning|warn|load_state" | tee $OUT/metrics.txt
echo RENDER_VAL_${NAME}_DONE
