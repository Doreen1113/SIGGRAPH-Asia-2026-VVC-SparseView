#!/bin/bash
#SBATCH --job-name=fgp_chain
#SBATCH --account=GOV114009
#SBATCH --partition=8gpus
#SBATCH --exclude=25a-hgpn144,25a-hgpn111
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --time=10:00:00
#SBATCH --output=/work/doreen071/vvc/jobs/fgp_chain_%j.log
# Person-region chain: render FGW20 on the test scenes -> person-source ensemble (6*SIX_ens + K*FGW20) ->
# Difix x3 TTA on it -> composite: person from FGP_TTA, background from SIX_pAVt15 (same as the 17:10 package),
# so the only difference vs that package is the person region.
module load cuda/12.6 2>/dev/null
export PATH=/home/doreen071/.conda/envs/vvc312/bin:$PATH TORCH_CUDA_ARCH_LIST=9.0
V=/work/doreen071/vvc; W=$V/work; PY=/home/doreen071/.conda/envs/vvc/bin/python
export OMP_NUM_THREADS=10 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "== waiting for FGW20_s0 on all 5 test scenes, $(date -u +%H:%M) =="
until (for c in 004_1_seq0 006_1_seq0 007_0_seq0 009_0_seq0 011_0_seq0; do [ -f $V/runs/siga_$c/run_FGW20_s0/gaussians.pt ] || exit 1; done); do sleep 60; done
echo "== rendering FGW20, $(date -u +%H:%M) =="
i=0
for c in 004_1_seq0 006_1_seq0 007_0_seq0 009_0_seq0 011_0_seq0; do
  OUT=$V/submissions/M_FGW20_s0/renders/$c; NEED=$(find $V/submissions/SIX_ens/renders/$c -name '*.jpg' | wc -l)
  [ "$(find $OUT -name '*.jpg' 2>/dev/null | wc -l)" -eq "$NEED" ] && { echo "complete $c"; continue; }
  ( TD=$(mktemp -d /tmp/tex.XXXXXX)/py312_cu121; mkdir -p $TD; cp -r $V/envcache/gsplat_prebuilt/gsplat_cuda $TD/
    TORCH_EXTENSIONS_DIR=$TD CUDA_VISIBLE_DEVICES=$i python $W/render_hidden.py $V/runs/siga_$c/run_FGW20_s0 $V/data/$c $OUT \
      --cull_near_frac 1.1 --cull_min_views 4 > $V/jobs/renderM_FGW20_$c.log 2>&1; rm -rf $(dirname $TD) ) &
  i=$((i+1))
done
wait
echo "FGW20 renders: $(find $V/submissions/M_FGW20_s0 -name '*.jpg' | wc -l)/2056"
K=$(cat $V/work/fgp_weight.txt 2>/dev/null || echo 3)
echo "== person-source ensemble 6*SIX + ${K}*FGW20, $(date -u +%H:%M) =="
$PY $W/weighted_ensemble.py $V/submissions/FGP_ens/renders $V/submissions/SIX_ens/renders:6 $V/submissions/M_FGW20_s0/renders:$K --procs 90
SRC=$V/submissions/FGP_ens/renders
echo "== difix 3 shifts on person source, $(date -u +%H:%M) =="
for s in "" "320,288" "640,192"; do
  case "$s" in "") vn=FGP_difix;; "320,288") vn=FGP_difix_s1;; *) vn=FGP_difix_s2;; esac
  for i in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES=$i DIFIX_SHARD=$i/8 $PY $W/difix_submission_h200.py $SRC $V/submissions/$vn/renders ${s:+--shift $s} > $V/jobs/fgp_${vn}_$i.log 2>&1 &
  done
  wait; echo "   $vn done $(date -u +%H:%M)"
done
$PY $W/tta_average.py $V/submissions/FGP_TTA/renders $V/submissions/FGP_difix/renders $V/submissions/FGP_difix_s1/renders $V/submissions/FGP_difix_s2/renders --procs 90
echo "== waiting for SIX_pAVt15 (background), $(date -u +%H:%M) =="
until [ "$(find $V/submissions/SIX_pAVt15 -name '*.jpg' 2>/dev/null | wc -l)" -eq 2056 ]; do sleep 120; done
echo "== composites: person <- FGP_TTA, background <- SIX_pAVt15, $(date -u +%H:%M) =="
i=0
for PA in 0.50 0.70; do
  TAG=$(printf "p%03d" $(echo "$PA*100/1" | bc))
  for c in 004_1_seq0 006_1_seq0 007_0_seq0 009_0_seq0 011_0_seq0; do
    CUDA_VISIBLE_DEVICES=$((i%8)) $PY $W/composite2.py tree $SRC $V/submissions/FGP_TTA/renders $PA \
      $V/submissions/SIX_pAVt15/renders 1.0 $V/submissions/FGP${K}_${TAG}_bpAVt15/renders --cases $c > $V/jobs/fgpcomp_${TAG}_$c.log 2>&1 &
    i=$((i+1)); [ $((i%8)) -eq 0 ] && wait
  done
done
wait
for PA in 050 070; do
  OUT=FGP${K}_p${PA}_bpAVt15; n=$(find $V/submissions/$OUT -name '*.jpg' 2>/dev/null | wc -l)
  echo "-- $OUT: $n/2056"; [ "$n" -eq 2056 ] && $PY $W/pack_submission.py $V/submissions/$OUT.zip $V/submissions/$OUT/renders
done
echo FGP_CHAIN_DONE $(date -u +%H:%M)
