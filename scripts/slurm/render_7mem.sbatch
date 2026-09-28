#!/bin/bash
#SBATCH --job-name=render_7mem
#SBATCH --account=GOV114009
#SBATCH --partition=dev
#SBATCH --exclude=25a-hgpn144,25a-hgpn111
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=60
#SBATCH --gres=gpu:5
#SBATCH --time=02:00:00
#SBATCH --output=/work/doreen071/vvc/jobs/render_7mem_%j.log
# 7-member ensemble render over all 5 test scenes: FULLRES_s0(main)+J_s0+GATE_s0+G4M_s0+G3M_FR_s0+GATE_FR_s0+FULLRES_s1, culled.
module load cuda/12.6 2>/dev/null
export PATH=/home/doreen071/.conda/envs/vvc312/bin:$PATH TORCH_CUDA_ARCH_LIST=9.0 OMP_NUM_THREADS=10
V=/work/doreen071/vvc
# wait until every member of the 7-member ensemble exists on all 5 test scenes
echo "waiting for all members..."
until for C in 004_1_seq0 006_1_seq0 007_0_seq0 009_0_seq0 011_0_seq0; do
        for M in FULLRES_s0 J_s0 GATE_s0 G4M_s0 G3M_FR_s0 GATE_FR_s0 FULLRES_s1; do
          [ -f $V/runs/siga_$C/run_$M/gaussians.pt ] || exit 1
        done
      done; do sleep 60; done
echo "all members present, starting render $(date -u +%H:%M)"
i=0
for c in 004_1_seq0 006_1_seq0 007_0_seq0 009_0_seq0 011_0_seq0; do
  R=$V/runs/siga_$c; OUT=$V/submissions/SEVEN_ens/renders/$c
  EX="$R/run_J_s0,$R/run_GATE_s0,$R/run_G4M_s0,$R/run_G3M_FR_s0,$R/run_GATE_FR_s0,$R/run_FULLRES_s1"
  CUDA_VISIBLE_DEVICES=$i python $V/work/render_hidden.py $R/run_FULLRES_s0 $V/data/$c $OUT \
    --cull_near_frac 1.1 --cull_min_views 4 --extra_runs "$EX" > $V/jobs/render7_$c.log 2>&1 &
  i=$((i+1))
done
wait
for c in 004_1_seq0 006_1_seq0 007_0_seq0 009_0_seq0 011_0_seq0; do echo "$c: $(find $V/submissions/SEVEN_ens/renders/$c -name '*.jpg' | wc -l)"; done
echo RENDER_7MEM_DONE
