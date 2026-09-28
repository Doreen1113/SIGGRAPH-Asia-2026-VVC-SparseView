#!/bin/bash
#SBATCH --job-name=fgep_chain
#SBATCH --account=GOV114009
#SBATCH --partition=8gpus
#SBATCH --exclude=25a-hgpn144,25a-hgpn111
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --time=08:00:00
#SBATCH --output=/work/doreen071/vvc/jobs/fgep_chain_%j.log
# Solo FG-SSIM on val: FGW40_L60K 0.8540, FGW20_s0 0.8412, FGW80_s0 0.8443, LONG60K_s0 0.8387. The current
# 6-member ensemble carries those three weak members, which is exactly the dilution val warned about. Build a
# PURE ensemble of the strongest config only (all 5 seeds of FGW40_L60K) and see if FG-SSIM comes back up.
V=/work/doreen071/vvc; W=$V/work; PY=/home/doreen071/.conda/envs/vvc/bin/python
export OMP_NUM_THREADS=10 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CASES="004_1_seq0 006_1_seq0 007_0_seq0 009_0_seq0 011_0_seq0"
# 5 seeds where available; tta_average skips an image a source lacks, so a second pass without s3 fills 006.
$PY $W/tta_average.py $V/submissions/FGEP_ens/renders \
  $V/submissions/M_FGW40_L60K/renders $V/submissions/M_FGW40_L60K_s1/renders $V/submissions/M_FGW40_L60K_s2/renders \
  $V/submissions/M_FGW40_L60K_s3/renders $V/submissions/M_FGW40_L60K_s4/renders --procs 90
$PY $W/tta_average.py $V/submissions/FGEP_ens/renders \
  $V/submissions/M_FGW40_L60K/renders $V/submissions/M_FGW40_L60K_s1/renders $V/submissions/M_FGW40_L60K_s2/renders \
  $V/submissions/M_FGW40_L60K_s4/renders --procs 90
echo "FGEP_ens: $(find $V/submissions/FGEP_ens -name '*.jpg' | wc -l)/2056"
for s in "" "320,288" "640,192"; do
  case "$s" in "") vn=FGEP_difix;; "320,288") vn=FGEP_difix_s1;; *) vn=FGEP_difix_s2;; esac
  for i in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES=$i DIFIX_SHARD=$i/8 $PY $W/difix_submission_h200.py $V/submissions/FGEP_ens/renders $V/submissions/$vn/renders ${s:+--shift $s} > $V/jobs/fgep_${vn}_$i.log 2>&1 &
  done; wait; echo "   $vn done $(date +%H:%M)"
done
$PY $W/tta_average.py $V/submissions/FGEP_TTA/renders $V/submissions/FGEP_difix/renders $V/submissions/FGEP_difix_s1/renders $V/submissions/FGEP_difix_s2/renders --procs 90
i=0
for PA in 0.85 0.70; do
  TAG=$(printf "p%03d" $(echo "$PA*100/1" | bc))
  for c in $CASES; do
    CUDA_VISIBLE_DEVICES=$((i%8)) $PY $W/composite2.py tree $V/submissions/FGEP_ens/renders $V/submissions/FGEP_TTA/renders $PA \
      $V/submissions/SIX_pAVt15/renders 1.0 $V/submissions/FGEP_${TAG}_bpAVt15/renders --cases $c > $V/jobs/fgepcomp_${TAG}_$c.log 2>&1 &
    i=$((i+1)); [ $((i%8)) -eq 0 ] && wait
  done
done
wait
for PA in 085 070; do
  OUT=FGEP_p${PA}_bpAVt15; n=$(find $V/submissions/$OUT -name '*.jpg' 2>/dev/null | wc -l)
  echo "-- $OUT: $n/2056"
  [ "$n" -eq 2056 ] && $PY $W/pack_submission.py $V/submissions/$OUT.zip $V/submissions/$OUT/renders && chmod 444 $V/submissions/$OUT.zip
done
echo FGEP_CHAIN_DONE $(date +%H:%M)
