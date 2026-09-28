#!/bin/bash
#SBATCH --job-name=build_nine
#SBATCH --account=GOV114009
#SBATCH --partition=8gpus
#SBATCH --exclude=25a-hgpn144,25a-hgpn111
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --time=08:00:00
#SBATCH --output=/work/doreen071/vvc/jobs/build_nine_%j.log
# Weighted 9/10-member ensemble -> Difix x3 TTA -> dual-backbone projection -> person/bg composite -> zip.
# Formula verified on val (mean |recon-direct| = 0.29/255): 6*SIX_ens folded in as the six original members.
V=/work/doreen071/vvc; W=$V/work; PY=/home/doreen071/.conda/envs/vvc/bin/python
export OMP_NUM_THREADS=10
MEMBERS="G3M_FR_s0 GATE_FR_s0 FULLRES_s1 G4M_FR_s0"
echo "== waiting for all new-member renders, $(date -u +%H:%M) =="
until (for M in $MEMBERS; do [ "$(find $V/submissions/M_$M -name '*.jpg' 2>/dev/null | wc -l)" -eq 2056 ] || exit 1; done); do sleep 60; done
echo "== weighted ensembles, $(date -u +%H:%M) =="
# 9-member: the val ladder's best mix (e7-equivalent: no G4M_FR, which was flat-to-worse as an 8th member)
$PY $W/weighted_ensemble.py $V/submissions/NINE_ens/renders \
  $V/submissions/SIX_ens/renders:6 $V/submissions/M_G3M_FR_s0/renders:1 \
  $V/submissions/M_GATE_FR_s0/renders:1 $V/submissions/M_FULLRES_s1/renders:1 --procs 90
SRC=$V/submissions/NINE_ens/renders
echo "== difix 3 shifts (8-way sharded), $(date -u +%H:%M) =="
for s in "" "320,288" "640,192"; do
  case "$s" in "") vn=NINE_difix;; "320,288") vn=NINE_difix_s1;; *) vn=NINE_difix_s2;; esac
  for i in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES=$i DIFIX_SHARD=$i/8 $PY $W/difix_submission_h200.py $SRC $V/submissions/$vn/renders ${s:+--shift $s} \
      > $V/jobs/nine_${vn}_shard$i.log 2>&1 &
  done
  wait; echo "   $vn done $(date -u +%H:%M)"
done
echo "== TTA average, $(date -u +%H:%M) =="
$PY $W/tta_average.py $V/submissions/NINE_TTA/renders $V/submissions/NINE_difix/renders $V/submissions/NINE_difix_s1/renders $V/submissions/NINE_difix_s2/renders --procs 90
echo "== dual-backbone projection lam15 (8-way sharded), $(date -u +%H:%M) =="
for i in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$i $PY $W/proj_multi.py tree $SRC $V/submissions/NINE_TTA/renders $V/submissions/NINE_pAVt15/renders \
    --nets alex:1,vgg:0.5 --lam 15 --steps 200 --batch 4 --shard $i/8 > $V/jobs/nine_proj_shard$i.log 2>&1 &
done
wait
echo "== composites (two candidates), $(date -u +%H:%M) =="
i=0
for c in 004_1_seq0 006_1_seq0 007_0_seq0 009_0_seq0 011_0_seq0; do
  CUDA_VISIBLE_DEVICES=$i $PY $W/composite2.py tree $SRC $V/submissions/NINE_TTA/renders 0.50 $V/submissions/NINE_pAVt15/renders 1.0 \
    $V/submissions/NINE_p050_bpAVt15/renders --cases $c > $V/jobs/nine_comp50_$c.log 2>&1 &
  CUDA_VISIBLE_DEVICES=$((i+5)) $PY $W/composite2.py tree $SRC $V/submissions/NINE_TTA/renders 0.60 $V/submissions/NINE_pAVt15/renders 1.0 \
    $V/submissions/NINE_p060_bpAVt15/renders --cases $c > $V/jobs/nine_comp60_$c.log 2>&1 &
  i=$((i+1))
  [ $i -eq 3 ] && wait && i=0
done
wait
for OUT in NINE_p050_bpAVt15 NINE_p060_bpAVt15; do
  echo "-- $OUT"
  for c in $(ls $SRC); do echo "   $c: $(find $V/submissions/$OUT/renders/$c -name '*.jpg' 2>/dev/null | wc -l) / $(find $SRC/$c -name '*.jpg' | wc -l)"; done
  $PY $W/pack_submission.py $V/submissions/$OUT.zip $V/submissions/$OUT/renders
done
echo BUILD_NINE_DONE $(date -u +%H:%M)
