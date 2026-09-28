#!/bin/bash
#SBATCH --job-name=fge6_chain
#SBATCH --account=GOV114009
#SBATCH --partition=8gpus
#SBATCH --exclude=25a-hgpn144,25a-hgpn111
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --time=10:00:00
#SBATCH --output=/work/doreen071/vvc/jobs/fge6_chain_%j.log
# val_fge2 showed that adding WEAKER configs dilutes the FG ensemble (9 members < 5 members). So instead of more
# configs we add more SEEDS of the strongest one (FGW40_L60K s1/s2 from job 380014): variance reduction without
# dilution. Person source = 6-member FG ensemble -> Difix x3 TTA -> alpha 0.85 over the SIX_pAVt15 background.
module load cuda/12.6 2>/dev/null
export PATH=/home/doreen071/.conda/envs/vvc312/bin:$PATH TORCH_CUDA_ARCH_LIST=9.0
V=/work/doreen071/vvc; W=$V/work; PY=/home/doreen071/.conda/envs/vvc/bin/python
export OMP_NUM_THREADS=10 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CASES="004_1_seq0 006_1_seq0 007_0_seq0 009_0_seq0 011_0_seq0"
echo "== waiting for FGW40_L60K seeds 1,2 on all 5 scenes, $(date +%H:%M) =="
until (for c in $CASES; do for s in s1 s2; do [ -f $V/runs/siga_$c/run_FGW40_L60K_$s/gaussians.pt ] || exit 1; done; done); do sleep 120; done
echo "== rendering the two new seeds, $(date +%H:%M) =="
i=0
for M in FGW40_L60K_s1 FGW40_L60K_s2; do
 for c in $CASES; do
  OUT=$V/submissions/M_$M/renders/$c; NEED=$(find $V/submissions/SIX_ens/renders/$c -name '*.jpg' | wc -l)
  [ "$(find $OUT -name '*.jpg' 2>/dev/null | wc -l)" -eq "$NEED" ] && { echo "complete $M/$c"; continue; }
  ( TD=$(mktemp -d /tmp/tex.XXXXXX)/py312_cu121; mkdir -p $TD; cp -r $V/envcache/gsplat_prebuilt/gsplat_cuda $TD/
    TORCH_EXTENSIONS_DIR=$TD CUDA_VISIBLE_DEVICES=$((i%8)) python $W/render_hidden.py $V/runs/siga_$c/run_$M $V/data/$c $OUT \
      --cull_near_frac 1.1 --cull_min_views 4 > $V/jobs/renderM_${M}_$c.log 2>&1; rm -rf $(dirname $TD) ) &
  i=$((i+1)); [ $((i%8)) -eq 0 ] && wait
 done
done
wait
for M in FGW40_L60K_s1 FGW40_L60K_s2; do echo "$M: $(find $V/submissions/M_$M -name '*.jpg' | wc -l)/2056"; done
echo "== 6-member FG ensemble, $(date +%H:%M) =="
$PY $W/tta_average.py $V/submissions/FGE6_ens/renders \
  $V/submissions/M_FGW40_L60K/renders $V/submissions/M_FGW40_L60K_s1/renders $V/submissions/M_FGW40_L60K_s2/renders \
  $V/submissions/M_FGW20_s0/renders $V/submissions/M_FGW80_s0/renders $V/submissions/M_LONG60K_s0/renders --procs 90
SRC=$V/submissions/FGE6_ens/renders
echo "== difix 3 shifts, $(date +%H:%M) =="
for s in "" "320,288" "640,192"; do
  case "$s" in "") vn=FGE6_difix;; "320,288") vn=FGE6_difix_s1;; *) vn=FGE6_difix_s2;; esac
  for i in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES=$i DIFIX_SHARD=$i/8 $PY $W/difix_submission_h200.py $SRC $V/submissions/$vn/renders ${s:+--shift $s} > $V/jobs/fge6_${vn}_$i.log 2>&1 &
  done; wait; echo "   $vn done $(date +%H:%M)"
done
$PY $W/tta_average.py $V/submissions/FGE6_TTA/renders $V/submissions/FGE6_difix/renders $V/submissions/FGE6_difix_s1/renders $V/submissions/FGE6_difix_s2/renders --procs 90
echo "== composites, $(date +%H:%M) =="
i=0
for PA in 0.85; do
  TAG=$(printf "p%03d" $(echo "$PA*100/1" | bc))
  for c in $CASES; do
    CUDA_VISIBLE_DEVICES=$((i%8)) $PY $W/composite2.py tree $SRC $V/submissions/FGE6_TTA/renders $PA \
      $V/submissions/SIX_pAVt15/renders 1.0 $V/submissions/FGE6_${TAG}_bpAVt15/renders --cases $c > $V/jobs/fge6comp_${TAG}_$c.log 2>&1 &
    i=$((i+1)); [ $((i%8)) -eq 0 ] && wait
  done
done
wait
OUT=FGE6_p085_bpAVt15; n=$(find $V/submissions/$OUT -name '*.jpg' 2>/dev/null | wc -l)
echo "-- $OUT: $n/2056"
[ "$n" -eq 2056 ] && $PY $W/pack_submission.py $V/submissions/$OUT.zip $V/submissions/$OUT/renders && chmod 444 $V/submissions/$OUT.zip
echo FGE6_CHAIN_DONE $(date +%H:%M)
