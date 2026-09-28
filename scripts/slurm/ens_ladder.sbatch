#!/bin/bash
#SBATCH --job-name=ens_ladder
#SBATCH --account=GOV114009
#SBATCH --partition=dev
#SBATCH --exclude=25a-hgpn144,25a-hgpn111
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:6
#SBATCH --time=02:00:00
#SBATCH --output=/work/doreen071/vvc/jobs/ens_ladder_%j.log
# Ensemble ladder on val 001_1: does adding members actually raise SSIM? Measured from scratch on H200-trained models.
module load cuda/12.6 2>/dev/null
export PATH=/home/doreen071/.conda/envs/vvc312/bin:$PATH
export TORCH_CUDA_ARCH_LIST=9.0
export TORCH_EXTENSIONS_DIR=$(mktemp -d /tmp/tex.XXXXXX)/py312_cu121
mkdir -p $TORCH_EXTENSIONS_DIR && cp -r /work/doreen071/vvc/envcache/gsplat_prebuilt/gsplat_cuda $TORCH_EXTENSIONS_DIR/
export OMP_NUM_THREADS=10
V=/work/doreen071/vvc; S=001_1_seq0; R=$V/runs/siga_$S
dump() { # gpu name extra_csv
  CUDA_VISIBLE_DEVICES=$1 python $V/work/dump_val_renders_h200.py $S $R/run_FULLRES_s0 "$3" $V/fixtest_h200/$S/$2 --every 200 \
    > $V/jobs/ens_$2.log 2>&1 &
}
dump 0 e1 ""
dump 1 e2 "$R/run_J_s0"
dump 2 e3 "$R/run_J_s0,$R/run_GATE_s0"
dump 3 e4 "$R/run_J_s0,$R/run_GATE_s0,$R/run_G4M_s0"
dump 4 e5 "$R/run_J_s0,$R/run_GATE_s0,$R/run_G4M_s0,$R/run_G3M_s0"
dump 5 e6 "$R/run_J_s0,$R/run_GATE_s0,$R/run_G4M_s0,$R/run_G3M_s0,$R/run_FULLRES_s1"
wait
for n in e1 e2 e3 e4 e5 e6; do
  echo "== $n ($(ls $V/fixtest_h200/$S/$n/*_render.png 2>/dev/null | wc -l) renders)"
  /home/doreen071/.conda/envs/vvc/bin/python $V/work/metrics_both.py $V/fixtest_h200/$S/$n render 2>&1 | grep -vE "Warning|warn|load_state"
done
echo ENS_LADDER_DONE
