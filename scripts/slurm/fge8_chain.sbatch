#!/bin/bash
#SBATCH --job-name=fge8_chain
#SBATCH --account=GOV114009
#SBATCH --partition=8gpus
#SBATCH --exclude=25a-hgpn144,25a-hgpn111
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --time=10:00:00
#SBATCH --output=/work/doreen071/vvc/jobs/fge8_chain_%j.log
# The 6-member FG ensemble took rank 1 (FINAL 2.167). FG-SSIM is 0.00008 short of one more rank (FINAL 2.000),
# so add two more seeds of the strongest config: 8 members, same recipe otherwise.
module load cuda/12.6 2>/dev/null
export PATH=/home/doreen071/.conda/envs/vvc312/bin:$PATH TORCH_CUDA_ARCH_LIST=9.0
V=/work/doreen071/vvc; W=$V/work; PY=/home/doreen071/.conda/envs/vvc/bin/python
export OMP_NUM_THREADS=10 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CASES="004_1_seq0 006_1_seq0 007_0_seq0 009_0_seq0 011_0_seq0"
echo "== waiting for seeds 3,4, $(date +%H:%M) =="
# s3 on 006 was killed by a 3:55 time limit at 94% and is retraining with a longer limit; do not block on it -
# build 006 from the 7 members it has and let the others use all 8. tta_average skips images a source lacks.
until (for c in $CASES; do [ -f $V/runs/siga_$c/run_FGW40_L60K_s4/gaussians.pt ] || exit 1; done); do sleep 120; done
echo "== rendering seeds 3,4, $(date +%H:%M) =="
i=0
for M in FGW40_L60K_s3 FGW40_L60K_s4; do
 for c in $CASES; do
  [ -f $V/runs/siga_$c/run_$M/gaussians.pt ] || { echo "no model yet: $M/$c"; continue; }
  OUT=$V/submissions/M_$M/renders/$c; NEED=$(find $V/submissions/SIX_ens/renders/$c -name '*.jpg' | wc -l)
  [ "$(find $OUT -name '*.jpg' 2>/dev/null | wc -l)" -eq "$NEED" ] && { echo "complete $M/$c"; continue; }
  ( TD=$(mktemp -d /tmp/tex.XXXXXX)/py312_cu121; mkdir -p $TD; cp -r $V/envcache/gsplat_prebuilt/gsplat_cuda $TD/
    TORCH_EXTENSIONS_DIR=$TD CUDA_VISIBLE_DEVICES=$((i%8)) python $W/render_hidden.py $V/runs/siga_$c/run_$M $V/data/$c $OUT \
      --cull_near_frac 1.1 --cull_min_views 4 > $V/jobs/renderM_${M}_$c.log 2>&1; rm -rf $(dirname $TD) ) &
  i=$((i+1)); [ $((i%8)) -eq 0 ] && wait
 done
done
wait
for M in FGW40_L60K_s3 FGW40_L60K_s4; do echo "$M: $(find $V/submissions/M_$M -name '*.jpg' | wc -l)/2056"; done
echo "== 8-member FG ensemble, $(date +%H:%M) =="
$PY $W/tta_average.py $V/submissions/FGE8_ens/renders \
  $V/submissions/M_FGW40_L60K/renders $V/submissions/M_FGW40_L60K_s1/renders $V/submissions/M_FGW40_L60K_s2/renders \
  $V/submissions/M_FGW40_L60K_s3/renders $V/submissions/M_FGW40_L60K_s4/renders \
  $V/submissions/M_FGW20_s0/renders $V/submissions/M_FGW80_s0/renders $V/submissions/M_LONG60K_s0/renders --procs 90
# second pass without s3 fills any scene whose s3 render is missing (006 while it retrains)
$PY $W/tta_average.py $V/submissions/FGE8_ens/renders \
  $V/submissions/M_FGW40_L60K/renders $V/submissions/M_FGW40_L60K_s1/renders $V/submissions/M_FGW40_L60K_s2/renders \
  $V/submissions/M_FGW40_L60K_s4/renders \
  $V/submissions/M_FGW20_s0/renders $V/submissions/M_FGW80_s0/renders $V/submissions/M_LONG60K_s0/renders --procs 90
echo "FGE8_ens: $(find $V/submissions/FGE8_ens -name '*.jpg' | wc -l)/2056"
echo FGE8_ENS_READY $(date +%H:%M)
# difix runs as three separate single-node jobs (launched below); wait for their output
until (for d in FGE8_difix FGE8_difix_s1 FGE8_difix_s2; do [ "$(find $V/submissions/$d -name '*.jpg' 2>/dev/null | wc -l)" -eq 2056 ] || exit 1; done); do sleep 60; done
$PY $W/tta_average.py $V/submissions/FGE8_TTA/renders $V/submissions/FGE8_difix/renders $V/submissions/FGE8_difix_s1/renders $V/submissions/FGE8_difix_s2/renders --procs 90
echo "== composites, $(date +%H:%M) =="
i=0
for PA in 0.85 0.92; do
  TAG=$(printf "p%03d" $(echo "$PA*100/1" | bc))
  for c in $CASES; do
    CUDA_VISIBLE_DEVICES=$((i%8)) $PY $W/composite2.py tree $V/submissions/FGE8_ens/renders $V/submissions/FGE8_TTA/renders $PA \
      $V/submissions/SIX_pAVt15/renders 1.0 $V/submissions/FGE8_${TAG}_bpAVt15/renders --cases $c > $V/jobs/fge8comp_${TAG}_$c.log 2>&1 &
    i=$((i+1)); [ $((i%8)) -eq 0 ] && wait
  done
done
wait
for PA in 085 092; do
  OUT=FGE8_p${PA}_bpAVt15; n=$(find $V/submissions/$OUT -name '*.jpg' 2>/dev/null | wc -l)
  echo "-- $OUT: $n/2056"
  [ "$n" -eq 2056 ] && $PY $W/pack_submission.py $V/submissions/$OUT.zip $V/submissions/$OUT/renders && chmod 444 $V/submissions/$OUT.zip
done
echo FGE8_CHAIN_DONE $(date +%H:%M)
