#!/bin/bash
#SBATCH --job-name=fgx_chain
#SBATCH --account=GOV114009
#SBATCH --partition=8gpus
#SBATCH --exclude=25a-hgpn144,25a-hgpn111
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --time=12:00:00
#SBATCH --output=/work/doreen071/vvc/jobs/fgx_chain_%j.log
# Measured trade-off on the real board:
#   pure peer person source     FG-SSIM 0.83203  FG-LPIPS 0.26493
#   3 seeds + 3 diverse (FGE6)  FG-SSIM 0.82914  FG-LPIPS 0.23779
#   5 seeds, one config (FGEP)  FG-SSIM 0.82250  FG-LPIPS 0.23295
# => DIVERSITY raises FG-SSIM, seed count lowers FG-LPIPS. So maximise both: 10 members across 4 configs,
# including the new 90K-iteration models (longer training was worth +0.014 FG-SSIM solo going 20K->60K).
V=/work/doreen071/vvc; W=$V/work; PY=/home/doreen071/.conda/envs/vvc/bin/python
export PATH=/home/doreen071/.conda/envs/vvc312/bin:$PATH TORCH_CUDA_ARCH_LIST=9.0
module load cuda/12.6 2>/dev/null
export OMP_NUM_THREADS=10 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CASES="004_1_seq0 006_1_seq0 007_0_seq0 009_0_seq0 011_0_seq0"
echo "== waiting for the 90K models, $(date +%H:%M) =="
until (for c in $CASES; do for M in FGW40_L90K FGW60_L90K; do [ -f $V/runs/siga_$c/run_$M/gaussians.pt ] || exit 1; done; done); do sleep 180; done
echo "== rendering them, $(date +%H:%M) =="
i=0
for M in FGW40_L90K FGW60_L90K; do
 for c in $CASES; do
  OUT=$V/submissions/M_$M/renders/$c; NEED=$(find $V/submissions/SIX_ens/renders/$c -name '*.jpg' | wc -l)
  [ "$(find $OUT -name '*.jpg' 2>/dev/null | wc -l)" -eq "$NEED" ] && continue
  ( TD=$(mktemp -d /tmp/tex.XXXXXX)/py312_cu121; mkdir -p $TD; cp -r $V/envcache/gsplat_prebuilt/gsplat_cuda $TD/
    TORCH_EXTENSIONS_DIR=$TD CUDA_VISIBLE_DEVICES=$((i%8)) python $W/render_hidden.py $V/runs/siga_$c/run_$M $V/data/$c $OUT \
      --cull_near_frac 1.1 --cull_min_views 4 > $V/jobs/renderM_${M}_$c.log 2>&1; rm -rf $(dirname $TD) ) &
  i=$((i+1)); [ $((i%8)) -eq 0 ] && wait
 done
done
wait
for M in FGW40_L90K FGW60_L90K; do echo "$M: $(find $V/submissions/M_$M -name '*.jpg' | wc -l)/2056"; done
echo "== 10-member diverse ensemble, $(date +%H:%M) =="
$PY $W/tta_average.py $V/submissions/FGX_ens/renders \
  $V/submissions/M_FGW40_L90K/renders $V/submissions/M_FGW60_L90K/renders \
  $V/submissions/M_FGW40_L60K/renders $V/submissions/M_FGW40_L60K_s1/renders $V/submissions/M_FGW40_L60K_s2/renders \
  $V/submissions/M_FGW40_L60K_s3/renders $V/submissions/M_FGW40_L60K_s4/renders \
  $V/submissions/M_FGW20_s0/renders $V/submissions/M_FGW80_s0/renders $V/submissions/M_LONG60K_s0/renders --procs 90
echo "FGX_ens: $(find $V/submissions/FGX_ens -name '*.jpg' | wc -l)/2056"
for s in "" "320,288" "640,192"; do
  case "$s" in "") vn=FGX_difix;; "320,288") vn=FGX_difix_s1;; *) vn=FGX_difix_s2;; esac
  for i in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES=$i DIFIX_SHARD=$i/8 $PY $W/difix_submission_h200.py $V/submissions/FGX_ens/renders $V/submissions/$vn/renders ${s:+--shift $s} > $V/jobs/fgx_${vn}_$i.log 2>&1 &
  done; wait; echo "   $vn done $(date +%H:%M)"
done
$PY $W/tta_average.py $V/submissions/FGX_TTA/renders $V/submissions/FGX_difix/renders $V/submissions/FGX_difix_s1/renders $V/submissions/FGX_difix_s2/renders --procs 90
i=0
for PA in 0.85 0.70; do
  TAG=$(printf "p%03d" $(echo "$PA*100/1" | bc))
  for c in $CASES; do
    CUDA_VISIBLE_DEVICES=$((i%8)) $PY $W/composite2.py tree $V/submissions/FGX_ens/renders $V/submissions/FGX_TTA/renders $PA \
      $V/submissions/SIX_pAVt15/renders 1.0 $V/submissions/FGX_${TAG}_bpAVt15/renders --cases $c > $V/jobs/fgxcomp_${TAG}_$c.log 2>&1 &
    i=$((i+1)); [ $((i%8)) -eq 0 ] && wait
  done
done
wait
for PA in 085 070; do
  OUT=FGX_p${PA}_bpAVt15; n=$(find $V/submissions/$OUT -name '*.jpg' 2>/dev/null | wc -l)
  echo "-- $OUT: $n/2056"
  [ "$n" -eq 2056 ] && $PY $W/pack_submission.py $V/submissions/$OUT.zip $V/submissions/$OUT/renders && chmod 444 $V/submissions/$OUT.zip
done
echo FGX_CHAIN_DONE $(date +%H:%M)
