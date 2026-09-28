#!/bin/bash
#SBATCH --job-name=dfxb_chain
#SBATCH --account=GOV114009
#SBATCH --partition=8gpus
#SBATCH --exclude=25a-hgpn144,25a-hgpn111
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --time=10:00:00
#SBATCH --output=/work/doreen071/vvc/jobs/dfxb_chain_%j.log
# DFX-in-the-base package. val: folding DFX into the PEER base at weight 2 improves ALL SIX metrics
# (PSNR +0.66, SSIM +0.0018, LPIPS -0.014, FG-LPIPS -0.036) - unlike our other members, which diluted it.
# base = 6*SIX_ens + 3*M_DFX_s0 -> Difix x3 TTA -> projection lam20 -> person composite (person source adds
# FGW80/LONG60K on top) -> zips.
V=/work/doreen071/vvc; W=$V/work; PY=/home/doreen071/.conda/envs/vvc/bin/python
export OMP_NUM_THREADS=10 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "== base ensemble 6*SIX + 3*DFX, $(date -u +%H:%M) =="
$PY $W/weighted_ensemble.py $V/submissions/DFXB_ens/renders $V/submissions/SIX_ens/renders:6 $V/submissions/M_DFX_s0/renders:3 --procs 90
SRC=$V/submissions/DFXB_ens/renders
echo "== difix 3 shifts, $(date -u +%H:%M) =="
for s in "" "320,288" "640,192"; do
  case "$s" in "") vn=DFXB_difix;; "320,288") vn=DFXB_difix_s1;; *) vn=DFXB_difix_s2;; esac
  for i in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES=$i DIFIX_SHARD=$i/8 $PY $W/difix_submission_h200.py $SRC $V/submissions/$vn/renders ${s:+--shift $s} > $V/jobs/dfxb_${vn}_$i.log 2>&1 &
  done
  wait; echo "   $vn done $(date -u +%H:%M)"
done
$PY $W/tta_average.py $V/submissions/DFXB_TTA/renders $V/submissions/DFXB_difix/renders $V/submissions/DFXB_difix_s1/renders $V/submissions/DFXB_difix_s2/renders --procs 90
echo "== projection lam20, $(date -u +%H:%M) =="
for i in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$i $PY $W/proj_multi.py tree $SRC $V/submissions/DFXB_TTA/renders $V/submissions/DFXB_pAVt20/renders \
    --nets alex:1,vgg:0.5 --lam 20 --steps 200 --batch 2 --shard $i/8 > $V/jobs/dfxb_proj_$i.log 2>&1 &
done
wait
echo "== person source = DFXB_ens + FGW80 + LONG60K, $(date -u +%H:%M) =="
$PY $W/weighted_ensemble.py $V/submissions/DFXP_ens/renders $V/submissions/DFXB_ens/renders:9 $V/submissions/M_FGW80_s0/renders:2 $V/submissions/M_LONG60K_s0/renders:2 --procs 90
for s in "" "320,288" "640,192"; do
  case "$s" in "") vn=DFXP_difix;; "320,288") vn=DFXP_difix_s1;; *) vn=DFXP_difix_s2;; esac
  for i in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES=$i DIFIX_SHARD=$i/8 $PY $W/difix_submission_h200.py $V/submissions/DFXP_ens/renders $V/submissions/$vn/renders ${s:+--shift $s} > $V/jobs/dfxp_${vn}_$i.log 2>&1 &
  done
  wait; echo "   $vn done $(date -u +%H:%M)"
done
$PY $W/tta_average.py $V/submissions/DFXP_TTA/renders $V/submissions/DFXP_difix/renders $V/submissions/DFXP_difix_s1/renders $V/submissions/DFXP_difix_s2/renders --procs 90
echo "== composites, $(date -u +%H:%M) =="
i=0
for PA in 0.40 0.50 0.65; do
  TAG=$(printf "p%03d" $(echo "$PA*100/1" | bc))
  for c in 004_1_seq0 006_1_seq0 007_0_seq0 009_0_seq0 011_0_seq0; do
    CUDA_VISIBLE_DEVICES=$((i%8)) $PY $W/composite2.py tree $V/submissions/DFXP_ens/renders $V/submissions/DFXP_TTA/renders $PA \
      $V/submissions/DFXB_pAVt20/renders 1.0 $V/submissions/DFXP_${TAG}_bpAVt20/renders --cases $c > $V/jobs/dfxpcomp_${TAG}_$c.log 2>&1 &
    i=$((i+1)); [ $((i%8)) -eq 0 ] && wait
  done
done
wait
for PA in 040 050 065; do
  OUT=DFXP_p${PA}_bpAVt20; n=$(find $V/submissions/$OUT -name '*.jpg' 2>/dev/null | wc -l)
  echo "-- $OUT: $n/2056"; [ "$n" -eq 2056 ] && $PY $W/pack_submission.py $V/submissions/$OUT.zip $V/submissions/$OUT/renders
done
echo DFXB_CHAIN_DONE $(date -u +%H:%M)
