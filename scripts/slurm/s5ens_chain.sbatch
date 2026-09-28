#!/bin/bash
#SBATCH --job-name=s5ens_chain
#SBATCH --account=GOV114009
#SBATCH --partition=8gpus
#SBATCH --exclude=25a-hgpn144,25a-hgpn111
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --time=10:00:00
#SBATCH --output=/work/doreen071/vvc/jobs/s5ens_chain_%j.log
# S5(點雲stride5)驗證顯示六項全面進步。渲染三個新種子,組成 S5 系列(4種子:原s0+新s1/s2/s3)強力ensemble,
# 加上其他互補設定(FGW20/FGW80/LONG60K)維持多樣性,取代舊的 FGW40_L60K 系列(stride10)。
module load cuda/12.6 2>/dev/null
export PATH=/home/doreen071/.conda/envs/vvc312/bin:$PATH TORCH_CUDA_ARCH_LIST=9.0
V=/work/doreen071/vvc; W=$V/work; PY=/home/doreen071/.conda/envs/vvc/bin/python
export OMP_NUM_THREADS=10 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CASES="004_1_seq0 006_1_seq0 007_0_seq0 009_0_seq0 011_0_seq0"
echo "== rendering S5 seeds 1,2,3 + S5 seed0(已在fixtest,但測試場景還沒render), $(date +%H:%M) =="
i=0
for M in S5_FGW40_L60K S5_FGW40_L60K_s1 S5_FGW40_L60K_s2 S5_FGW40_L60K_s3; do
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
for M in S5_FGW40_L60K S5_FGW40_L60K_s1 S5_FGW40_L60K_s2 S5_FGW40_L60K_s3; do echo "$M: $(find $V/submissions/M_$M -name '*.jpg' | wc -l)/2056"; done
echo "== S5 系列(4種子)+ 互補設定, ensemble, $(date +%H:%M) =="
$PY $W/tta_average.py $V/submissions/S5E_ens/renders \
  $V/submissions/M_S5_FGW40_L60K/renders $V/submissions/M_S5_FGW40_L60K_s1/renders \
  $V/submissions/M_S5_FGW40_L60K_s2/renders $V/submissions/M_S5_FGW40_L60K_s3/renders \
  $V/submissions/M_FGW20_s0/renders $V/submissions/M_FGW80_s0/renders $V/submissions/M_LONG60K_s0/renders --procs 90
echo "S5E_ens: $(find $V/submissions/S5E_ens -name '*.jpg' | wc -l)/2056"
for s in "" "320,288" "640,192"; do
  case "$s" in "") vn=S5E_difix;; "320,288") vn=S5E_difix_s1;; *) vn=S5E_difix_s2;; esac
  for i in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES=$i DIFIX_SHARD=$i/8 $PY $W/difix_submission_h200.py $V/submissions/S5E_ens/renders $V/submissions/$vn/renders ${s:+--shift $s} > $V/jobs/s5e_${vn}_$i.log 2>&1 &
  done; wait; echo "   $vn done $(date +%H:%M)"
done
$PY $W/tta_average.py $V/submissions/S5E_TTA/renders $V/submissions/S5E_difix/renders $V/submissions/S5E_difix_s1/renders $V/submissions/S5E_difix_s2/renders --procs 90
echo "== composites, $(date +%H:%M) =="
i=0
for PA in 0.85 0.70; do
  TAG=$(printf "p%03d" $(echo "$PA*100/1" | bc))
  for c in $CASES; do
    CUDA_VISIBLE_DEVICES=$((i%8)) $PY $W/composite2.py tree $V/submissions/S5E_ens/renders $V/submissions/S5E_TTA/renders $PA \
      $V/submissions/SIX_pAVt15/renders 1.0 $V/submissions/S5E_${TAG}_bpAVt15/renders --cases $c > $V/jobs/s5ecomp_${TAG}_$c.log 2>&1 &
    i=$((i+1)); [ $((i%8)) -eq 0 ] && wait
  done
done
wait
for PA in 085 070; do
  OUT=S5E_p${PA}_bpAVt15; n=$(find $V/submissions/$OUT -name '*.jpg' 2>/dev/null | wc -l)
  echo "-- $OUT: $n/2056"
  [ "$n" -eq 2056 ] && $PY $W/pack_submission.py $V/submissions/$OUT.zip $V/submissions/$OUT/renders && chmod 444 $V/submissions/$OUT.zip
done
echo S5ENS_CHAIN_DONE $(date +%H:%M)
