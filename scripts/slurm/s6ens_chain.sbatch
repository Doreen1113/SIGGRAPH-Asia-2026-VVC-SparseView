#!/bin/bash
#SBATCH --job-name=s6ens_chain
#SBATCH --account=GOV114009
#SBATCH --partition=8gpus
#SBATCH --exclude=25a-hgpn144,25a-hgpn111
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --time=10:00:00
#SBATCH --output=/work/doreen071/vvc/jobs/s6ens_chain_%j.log
# S5E(4種子)拿到FINAL 2.667史上最佳。現在用全部6個S5種子(s0-s5)+互補設定,看能否再往上推。
module load cuda/12.6 2>/dev/null
export PATH=/home/doreen071/.conda/envs/vvc312/bin:$PATH TORCH_CUDA_ARCH_LIST=9.0
V=/work/doreen071/vvc; W=$V/work; PY=/home/doreen071/.conda/envs/vvc/bin/python
export OMP_NUM_THREADS=10 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CASES="004_1_seq0 006_1_seq0 007_0_seq0 009_0_seq0 011_0_seq0"
echo "== rendering S5 seeds 4,5, $(date +%H:%M) =="
i=0
for M in S5_FGW40_L60K_s4 S5_FGW40_L60K_s5; do
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
for M in S5_FGW40_L60K_s4 S5_FGW40_L60K_s5; do echo "$M: $(find $V/submissions/M_$M -name '*.jpg' | wc -l)/2056"; done
echo "== 6種子S5 + 互補設定, ensemble, $(date +%H:%M) =="
$PY $W/tta_average.py $V/submissions/S6E_ens/renders \
  $V/submissions/M_S5_FGW40_L60K/renders $V/submissions/M_S5_FGW40_L60K_s1/renders \
  $V/submissions/M_S5_FGW40_L60K_s2/renders $V/submissions/M_S5_FGW40_L60K_s3/renders \
  $V/submissions/M_S5_FGW40_L60K_s4/renders $V/submissions/M_S5_FGW40_L60K_s5/renders \
  $V/submissions/M_FGW20_s0/renders $V/submissions/M_FGW80_s0/renders $V/submissions/M_LONG60K_s0/renders --procs 90
echo "S6E_ens: $(find $V/submissions/S6E_ens -name '*.jpg' | wc -l)/2056"
for s in "" "320,288" "640,192"; do
  case "$s" in "") vn=S6E_difix;; "320,288") vn=S6E_difix_s1;; *) vn=S6E_difix_s2;; esac
  for i in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES=$i DIFIX_SHARD=$i/8 $PY $W/difix_submission_h200.py $V/submissions/S6E_ens/renders $V/submissions/$vn/renders ${s:+--shift $s} > $V/jobs/s6e_${vn}_$i.log 2>&1 &
  done; wait; echo "   $vn done $(date +%H:%M)"
done
$PY $W/tta_average.py $V/submissions/S6E_TTA/renders $V/submissions/S6E_difix/renders $V/submissions/S6E_difix_s1/renders $V/submissions/S6E_difix_s2/renders --procs 90
echo "== composites, $(date +%H:%M) =="
i=0
for PA in 0.85 0.70; do
  TAG=$(printf "p%03d" $(echo "$PA*100/1" | bc))
  for c in $CASES; do
    CUDA_VISIBLE_DEVICES=$((i%8)) $PY $W/composite2.py tree $V/submissions/S6E_ens/renders $V/submissions/S6E_TTA/renders $PA \
      $V/submissions/SIX_pAVt15/renders 1.0 $V/submissions/S6E_${TAG}_bpAVt15/renders --cases $c > $V/jobs/s6ecomp_${TAG}_$c.log 2>&1 &
    i=$((i+1)); [ $((i%8)) -eq 0 ] && wait
  done
done
wait
for PA in 085 070; do
  OUT=S6E_p${PA}_bpAVt15; n=$(find $V/submissions/$OUT -name '*.jpg' 2>/dev/null | wc -l)
  echo "-- $OUT: $n/2056"
  [ "$n" -eq 2056 ] && $PY $W/pack_submission.py $V/submissions/$OUT.zip $V/submissions/$OUT/renders && chmod 444 $V/submissions/$OUT.zip
done
echo S6ENS_CHAIN_DONE $(date +%H:%M)
