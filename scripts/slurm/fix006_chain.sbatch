#!/bin/bash
#SBATCH --job-name=fix006
#SBATCH --account=GOV114009
#SBATCH --partition=8gpus
#SBATCH --exclude=25a-hgpn144,25a-hgpn111
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH --time=02:00:00
#SBATCH --output=/work/doreen071/vvc/jobs/fix006_%j.log
# FGW40_L60K_s3's model for scene 006 finished training AFTER the FGE8 render step had already moved past it,
# so it was never rendered -> FGX_ens/renders/006_1_seq0 never got built -> FGX composite crashed on that scene.
# Render it now, then let the already-resumable ensemble/difix/composite steps backfill just scene 006.
V=/work/doreen071/vvc; W=$V/work; PY=/home/doreen071/.conda/envs/vvc/bin/python
export PATH=/home/doreen071/.conda/envs/vvc312/bin:$PATH TORCH_CUDA_ARCH_LIST=9.0
module load cuda/12.6 2>/dev/null
echo "== render s3/006, $(date +%H:%M) =="
TD=$(mktemp -d /tmp/tex.XXXXXX)/py312_cu121; mkdir -p $TD; cp -r $V/envcache/gsplat_prebuilt/gsplat_cuda $TD/
TORCH_EXTENSIONS_DIR=$TD CUDA_VISIBLE_DEVICES=0 python $W/render_hidden.py $V/runs/siga_006_1_seq0/run_FGW40_L60K_s3 $V/data/006_1_seq0 $V/submissions/M_FGW40_L60K_s3/renders/006_1_seq0 \
  --cull_near_frac 1.1 --cull_min_views 4 > $V/jobs/renderM_FGW40_L60K_s3_006_1_seq0_fix.log 2>&1
rm -rf $(dirname $TD)
n=$(find $V/submissions/M_FGW40_L60K_s3/renders/006_1_seq0 -name '*.jpg' | wc -l)
echo "M_FGW40_L60K_s3/006: $n/440"
export OMP_NUM_THREADS=10 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "== backfill FGX_ens for 006, $(date +%H:%M) =="
$PY $W/tta_average.py $V/submissions/FGX_ens/renders \
  $V/submissions/M_FGW40_L90K/renders $V/submissions/M_FGW60_L90K/renders \
  $V/submissions/M_FGW40_L60K/renders $V/submissions/M_FGW40_L60K_s1/renders $V/submissions/M_FGW40_L60K_s2/renders \
  $V/submissions/M_FGW40_L60K_s3/renders $V/submissions/M_FGW40_L60K_s4/renders \
  $V/submissions/M_FGW20_s0/renders $V/submissions/M_FGW80_s0/renders $V/submissions/M_LONG60K_s0/renders --procs 90
n=$(find $V/submissions/FGX_ens/renders/006_1_seq0 -name '*.jpg' 2>/dev/null | wc -l)
echo "FGX_ens/006: $n/416"
echo "== backfill difix x3 for 006, $(date +%H:%M) =="
for s in "" "320,288" "640,192"; do
  case "$s" in "") vn=FGX_difix;; "320,288") vn=FGX_difix_s1;; *) vn=FGX_difix_s2;; esac
  for i in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES=$i DIFIX_SHARD=$i/8 $PY $W/difix_submission_h200.py $V/submissions/FGX_ens/renders $V/submissions/$vn/renders ${s:+--shift $s} > $V/jobs/fgxfix_${vn}_$i.log 2>&1 &
  done; wait
  echo "   $vn: $(find $V/submissions/$vn/renders/006_1_seq0 -name '*.jpg' 2>/dev/null | wc -l)/416"
done
$PY $W/tta_average.py $V/submissions/FGX_TTA/renders $V/submissions/FGX_difix/renders $V/submissions/FGX_difix_s1/renders $V/submissions/FGX_difix_s2/renders --procs 90
echo "== backfill composite for 006, $(date +%H:%M) =="
for PA in 0.85 0.70; do
  TAG=$(printf "p%03d" $(echo "$PA*100/1" | bc))
  CUDA_VISIBLE_DEVICES=0 $PY $W/composite2.py tree $V/submissions/FGX_ens/renders $V/submissions/FGX_TTA/renders $PA \
    $V/submissions/SIX_pAVt15/renders 1.0 $V/submissions/FGX_${TAG}_bpAVt15/renders --cases 006_1_seq0 > $V/jobs/fgxfixcomp_$TAG.log 2>&1
done
for PA in 085 070; do
  OUT=FGX_p${PA}_bpAVt15; n=$(find $V/submissions/$OUT -name '*.jpg' 2>/dev/null | wc -l)
  echo "-- $OUT: $n/2056"
  [ "$n" -eq 2056 ] && rm -f $V/submissions/$OUT.zip && $PY $W/pack_submission.py $V/submissions/$OUT.zip $V/submissions/$OUT/renders && chmod 444 $V/submissions/$OUT.zip
done
echo FIX006_DONE $(date +%H:%M)
