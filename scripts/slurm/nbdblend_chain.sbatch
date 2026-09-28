#!/bin/bash
#SBATCH --job-name=nbdblend_chain
#SBATCH --account=GOV114009
#SBATCH --partition=8gpus
#SBATCH --exclude=25a-hgpn144,25a-hgpn111
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --time=06:00:00
#SBATCH --output=/work/doreen071/vvc/jobs/nbdblend_chain_%j.log
# NBD_S5_FG (distillation+stride5+FG weight) alone loses FG-SSIM vs pure S5 (-0.0035). But diversity + Difix TTA
# has repeatedly RAISED FG-SSIM when blending different techniques (val_personsrc: half S5 + diverse mix = +0.0046
# to +0.0071 FG-SSIM, all-six better). Test the same idea with NBD_S5_FG as the new diverse ingredient:
#   person source = mean(6x S5 seeds, FGW20_s0, FGW80_s0, LONG60K_s0, NBD_S5_FG)  -- NBD_S5_FG diluted at weight 1
#   among 9 others, so any of its own weaknesses get averaged out while its distillation-quality texture helps.
# Background = NBX15_pAV (already proven all-six-better on val, and the live board's current best already uses it).
V=/work/doreen071/vvc; W=$V/work; PY=/home/doreen071/.conda/envs/vvc/bin/python
export PATH=/home/doreen071/.conda/envs/vvc312/bin:$PATH TORCH_CUDA_ARCH_LIST=9.0
module load cuda/12.6 2>/dev/null
export OMP_NUM_THREADS=10 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CASES="004_1_seq0 006_1_seq0 007_0_seq0 009_0_seq0 011_0_seq0"
echo "== render NBD_S5_FG, $(date +%H:%M) =="
i=0
for c in $CASES; do
  OUT=$V/submissions/M_NBD_S5_FG/renders/$c; NEED=$(find $V/submissions/SIX_pAVt15/renders/$c -name '*.jpg' | wc -l)
  [ "$(find $OUT -name '*.jpg' 2>/dev/null | wc -l)" -eq "$NEED" ] && { echo "complete $c"; continue; }
  ( TD=$(mktemp -d /tmp/tex.XXXXXX)/py312_cu121; mkdir -p $TD; cp -r $V/envcache/gsplat_prebuilt/gsplat_cuda $TD/
    TORCH_EXTENSIONS_DIR=$TD CUDA_VISIBLE_DEVICES=$((i%8)) python $W/render_hidden.py $V/runs/siga_$c/run_NBD_S5_FG $V/data/$c $OUT \
      --cull_near_frac 1.1 --cull_min_views 4 > $V/jobs/renderM_NBD_S5_FG_$c.log 2>&1; rm -rf $(dirname $TD) ) &
  i=$((i+1))
done
wait
echo "M_NBD_S5_FG: $(find $V/submissions/M_NBD_S5_FG -name '*.jpg' | wc -l)/2056"
echo "== build blended person source, $(date +%H:%M) =="
$PY $W/tta_average.py $V/submissions/NBDB_ens/renders \
  $V/submissions/M_S5_FGW40_L60K/renders $V/submissions/M_S5_FGW40_L60K_s1/renders $V/submissions/M_S5_FGW40_L60K_s2/renders \
  $V/submissions/M_S5_FGW40_L60K_s3/renders $V/submissions/M_S5_FGW40_L60K_s4/renders $V/submissions/M_S5_FGW40_L60K_s5/renders \
  $V/submissions/M_FGW20_s0/renders $V/submissions/M_FGW80_s0/renders $V/submissions/M_LONG60K_s0/renders \
  $V/submissions/M_NBD_S5_FG/renders --procs 90
SRC=$V/submissions/NBDB_ens/renders
echo "NBDB_ens: $(find $SRC -name '*.jpg' | wc -l)/2056"
echo "== difix x3, $(date +%H:%M) =="
for s in "" "320,288" "640,192"; do
  case "$s" in "") vn=NBDB_difix;; "320,288") vn=NBDB_difix_s1;; *) vn=NBDB_difix_s2;; esac
  for i in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES=$i DIFIX_SHARD=$i/8 $PY $W/difix_submission_h200.py $SRC $V/submissions/$vn/renders ${s:+--shift $s} > $V/jobs/nbdb_${vn}_$i.log 2>&1 &
  done; wait; echo "   $vn done $(date +%H:%M)"
done
$PY $W/tta_average.py $V/submissions/NBDB_TTA/renders $V/submissions/NBDB_difix/renders $V/submissions/NBDB_difix_s1/renders $V/submissions/NBDB_difix_s2/renders --procs 90
echo "== composites (alpha 0.85 / 0.70), $(date +%H:%M) =="
i=0
for PA in 0.85 0.70; do
  TAG=$(printf "p%03d" $(echo "$PA*100/1" | bc))
  for c in $CASES; do
    CUDA_VISIBLE_DEVICES=$((i%8)) $PY $W/composite2.py tree $SRC $V/submissions/NBDB_TTA/renders $PA \
      $V/submissions/NBX15_pAV/renders 1.0 $V/submissions/NBDB_${TAG}/renders --cases $c > $V/jobs/nbdbcomp_${TAG}_$c.log 2>&1 &
    i=$((i+1)); [ $((i%8)) -eq 0 ] && wait
  done
done
wait
for PA in 085 070; do
  OUT=NBDB_p${PA}; n=$(find $V/submissions/$OUT -name '*.jpg' 2>/dev/null | wc -l)
  echo "-- $OUT: $n/2056"
  [ "$n" -eq 2056 ] && $PY $W/pack_submission.py $V/submissions/$OUT.zip $V/submissions/$OUT/renders && chmod 444 $V/submissions/$OUT.zip
done
echo NBDBLEND_CHAIN_DONE $(date +%H:%M)
