#!/bin/bash
#SBATCH --job-name=dfxp_fin
#SBATCH --account=GOV114009
#SBATCH --partition=8gpus
#SBATCH --exclude=25a-hgpn144,25a-hgpn111
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --time=08:00:00
#SBATCH --output=/work/doreen071/vvc/jobs/dfxp_fin_%j.log
# Person source on top of the DFX base, then composites against the DFXB projection background.
# Person members: FGW40_L60K if it has finished on all 5 test scenes (val: best FG model by a wide margin),
# otherwise fall back to FGW80 + LONG60K.
V=/work/doreen071/vvc; W=$V/work; PY=/home/doreen071/.conda/envs/vvc/bin/python
export OMP_NUM_THREADS=10 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CASES="004_1_seq0 006_1_seq0 007_0_seq0 009_0_seq0 011_0_seq0"
echo "== waiting up to 2h for FGW40_L60K on all 5 test scenes, $(date -u +%H:%M) =="
END=$(( $(date +%s) + 7200 ))
while [ $(date +%s) -lt $END ]; do
  ok=1; for c in $CASES; do [ -f $V/runs/siga_$c/run_FGW40_L60K/gaussians.pt ] || ok=0; done
  [ $ok -eq 1 ] && break; sleep 120
done
PERSON_ARGS=""
ok=1; for c in $CASES; do [ -f $V/runs/siga_$c/run_FGW40_L60K/gaussians.pt ] || ok=0; done
if [ $ok -eq 1 ]; then
  echo "using FGW40_L60K as the person member"
  module load cuda/12.6 2>/dev/null
  export PATH=/home/doreen071/.conda/envs/vvc312/bin:$PATH TORCH_CUDA_ARCH_LIST=9.0
  i=0
  for c in $CASES; do
    OUT=$V/submissions/M_FGW40_L60K/renders/$c; NEED=$(find $V/submissions/SIX_ens/renders/$c -name '*.jpg' | wc -l)
    [ "$(find $OUT -name '*.jpg' 2>/dev/null | wc -l)" -eq "$NEED" ] && continue
    ( TD=$(mktemp -d /tmp/tex.XXXXXX)/py312_cu121; mkdir -p $TD; cp -r $V/envcache/gsplat_prebuilt/gsplat_cuda $TD/
      TORCH_EXTENSIONS_DIR=$TD CUDA_VISIBLE_DEVICES=$i python $W/render_hidden.py $V/runs/siga_$c/run_FGW40_L60K $V/data/$c $OUT \
        --cull_near_frac 1.1 --cull_min_views 4 > $V/jobs/renderM_FGW40_L60K_$c.log 2>&1; rm -rf $(dirname $TD) ) &
    i=$((i+1))
  done
  wait
  echo "FGW40_L60K renders: $(find $V/submissions/M_FGW40_L60K -name '*.jpg' | wc -l)/2056"
  PERSON_ARGS="$V/submissions/M_FGW40_L60K/renders:4"
  TAGBASE=DFXP4
else
  echo "FGW40_L60K not ready, falling back to FGW80 + LONG60K"
  PERSON_ARGS="$V/submissions/M_FGW80_s0/renders:2 $V/submissions/M_LONG60K_s0/renders:2"
  TAGBASE=DFXP8
fi
echo "== person-source ensemble, $(date -u +%H:%M) =="
$PY $W/weighted_ensemble.py $V/submissions/${TAGBASE}_ens/renders $V/submissions/DFXB_ens/renders:9 $PERSON_ARGS --procs 90
for s in "" "320,288" "640,192"; do
  case "$s" in "") vn=${TAGBASE}_difix;; "320,288") vn=${TAGBASE}_difix_s1;; *) vn=${TAGBASE}_difix_s2;; esac
  for i in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES=$i DIFIX_SHARD=$i/8 $PY $W/difix_submission_h200.py $V/submissions/${TAGBASE}_ens/renders $V/submissions/$vn/renders ${s:+--shift $s} > $V/jobs/${TAGBASE}_${vn}_$i.log 2>&1 &
  done
  wait; echo "   $vn done $(date -u +%H:%M)"
done
$PY $W/tta_average.py $V/submissions/${TAGBASE}_TTA/renders $V/submissions/${TAGBASE}_difix/renders $V/submissions/${TAGBASE}_difix_s1/renders $V/submissions/${TAGBASE}_difix_s2/renders --procs 90
echo "== waiting for the DFXB projection background, $(date -u +%H:%M) =="
until [ "$(find $V/submissions/DFXB_pAVt20 -name '*.jpg' 2>/dev/null | wc -l)" -eq 2056 ]; do sleep 120; done
echo "== composites, $(date -u +%H:%M) =="
i=0
for PA in 0.40 0.50 0.65; do
  TAG=$(printf "p%03d" $(echo "$PA*100/1" | bc))
  for c in $CASES; do
    CUDA_VISIBLE_DEVICES=$((i%8)) $PY $W/composite2.py tree $V/submissions/${TAGBASE}_ens/renders $V/submissions/${TAGBASE}_TTA/renders $PA \
      $V/submissions/DFXB_pAVt20/renders 1.0 $V/submissions/${TAGBASE}_${TAG}_bpAVt20/renders --cases $c > $V/jobs/${TAGBASE}comp_${TAG}_$c.log 2>&1 &
    i=$((i+1)); [ $((i%8)) -eq 0 ] && wait
  done
done
wait
for PA in 040 050 065; do
  OUT=${TAGBASE}_p${PA}_bpAVt20; n=$(find $V/submissions/$OUT -name '*.jpg' 2>/dev/null | wc -l)
  echo "-- $OUT: $n/2056"; [ "$n" -eq 2056 ] && $PY $W/pack_submission.py $V/submissions/$OUT.zip $V/submissions/$OUT/renders
done
echo DFXP_FINISH_DONE $(date -u +%H:%M)
