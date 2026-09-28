#!/bin/bash
#SBATCH --job-name=render_test
#SBATCH --account=GOV114009
#SBATCH --partition=dev
#SBATCH --exclude=25a-hgpn144,25a-hgpn111
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=60
#SBATCH --gres=gpu:5
#SBATCH --time=02:00:00
#SBATCH --output=/work/doreen071/vvc/jobs/render_test_%j.log
# Usage: sbatch --export=ALL,MEMBER=GATE_FR_s0 render_test.sbatch  -> submissions/M_<MEMBER>/renders/<case>/<view>/*.jpg (culled, single member)
module load cuda/12.6 2>/dev/null
export PATH=/home/doreen071/.conda/envs/vvc312/bin:$PATH TORCH_CUDA_ARCH_LIST=9.0 OMP_NUM_THREADS=8
V=/work/doreen071/vvc; i=0
for c in 004_1_seq0 006_1_seq0 007_0_seq0 009_0_seq0 011_0_seq0; do
export  OUT=$V/submissions/M_$MEMBER/renders/$c
export  CUDA_VISIBLE_DEVICES=$i python $V/work/render_hidden.py $V/runs/siga_$c/run_$MEMBER $V/data/$c $OUT --cull_near_frac 1.1 --cull_min_views 4 > $V/jobs/render_${MEMBER}_$c.log 2>&1 &
  i=$((i+1))
done
wait
for c in 004_1_seq0 006_1_seq0 007_0_seq0 009_0_seq0 011_0_seq0; do echo "$c: $(find $V/submissions/M_$MEMBER/renders/$c -name '*.jpg' | wc -l)"; done
echo RENDER_TEST_${MEMBER}_DONE
