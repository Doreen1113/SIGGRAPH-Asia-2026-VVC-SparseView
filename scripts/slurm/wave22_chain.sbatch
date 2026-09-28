#!/bin/bash
#SBATCH --job-name=w22_chain
#SBATCH --account=GOV114009
#SBATCH --partition=8gpus
#SBATCH --exclude=25a-hgpn144,25a-hgpn111
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --time=06:00:00
#SBATCH --output=/work/doreen071/vvc/jobs/w22_chain_%j.log
# Waits for wave22 (S5PS/S5PS_s1/S5PS_w30: stride-5 keyframes x SEVA pseudo-view supervision, the first time these
# two proven levers have been combined) to finish training, then folds it into the existing best person source
# as a 3-seed addition (weight 3, matching how S5P_ens already folds in S5_FGW40_L60K seeds) -> rebuild
# NB3/HB3/Difix/composite -> C9 package. Colour-correction (C8's win) is layered on top afterward if C8 confirms.
module load cuda/12.6 2>/dev/null
V=/work/doreen071/vvc; W=$V/work; S=$V/submissions; PY=/home/doreen071/.conda/envs/vvc/bin/python
export OMP_NUM_THREADS=10 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cnt() { find $1 -name '*.jpg' 2>/dev/null | wc -l; }
CASES="004_1_seq0 006_1_seq0 007_0_seq0 009_0_seq0 011_0_seq0"
for m in S5PS S5PS_s1 S5PS_w30; do
  for c in $CASES; do until [ -f $V/runs/siga_$c/run_$m/gaussians.pt ]; do sleep 60; done; done
done
echo "== wave22 train done, render, $(date +%H:%M) =="
render() {
  OUT=$S/M_$1/renders/$2
  [ "$(cnt $OUT)" -gt 0 ] && return
  TD=$(mktemp -d /tmp/tex.XXXXXX)/py312_cu121; mkdir -p $TD; cp -r $V/envcache/gsplat_prebuilt/gsplat_cuda $TD/
  PATH=/home/doreen071/.conda/envs/vvc312/bin:$PATH TORCH_CUDA_ARCH_LIST=9.0 TORCH_EXTENSIONS_DIR=$TD CUDA_VISIBLE_DEVICES=$3 python $W/render_hidden.py $V/runs/siga_$2/run_$1 $V/data/$2 $OUT --cull_near_frac 1.1 --cull_min_views 4 > $V/jobs/renderM_$1_$2.log 2>&1
  rm -rf $(dirname $TD)
}
k=0; JOBS=()
for m in S5PS S5PS_s1 S5PS_w30; do for c in $CASES; do JOBS+=("$m $c"); done; done
for j in "${JOBS[@]}"; do render $j $((k % 8)) & k=$((k+1)); [ $((k % 8)) -eq 0 ] && wait; done; wait
for m in S5PS S5PS_s1 S5PS_w30; do echo "M_$m: $(cnt $S/M_$m)/2056"; [ "$(cnt $S/M_$m)" -eq 2056 ] || { echo "RENDER INCOMPLETE $m"; exit 1; }; done
echo "== S5P_v2 (9-seed) / NB3v4 / HB3v4, $(date +%H:%M) =="
# S5P_ens is currently a 6-seed mean of S5_FGW40_L60K(_s1..s5) with equal weight; fold in the 3 new S5PS variants
# at equal per-render weight (3 members, each already an average over its own frames) alongside the existing 6.
$PY $W/weighted_ensemble.py $S/S5Pv2_ens/renders $S/S5P_ens/renders:6 $S/M_S5PS/renders:1 $S/M_S5PS_s1/renders:1 $S/M_S5PS_w30/renders:1 --procs 90
$PY $W/weighted_ensemble.py $S/NB3v4_ens/renders $S/SIX_ens/renders:4 $S/M_DFX_s0/renders:2 $S/M_DFXFG_L60K/renders:1 $S/M_DFXFG_L60K_s1/renders:1 $S/S5Pv2_ens/renders:2 $S/M_S3_FGW40_L60K/renders:2 --procs 90
$PY $W/tta_average.py $S/HB3v4_ens/renders $S/S5Pv2_ens/renders $S/NB3v4_ens/renders --procs 90
echo "HB3v4_ens: $(cnt $S/HB3v4_ens)/2056"
for s in "" "320,288" "640,192"; do
  case "$s" in "") vn=HB3v4_difix;; "320,288") vn=HB3v4_difix_s1;; *) vn=HB3v4_difix_s2;; esac
  echo "== difix $vn, $(date +%H:%M) =="
  for i in 0 1 2 3 4 5 6 7; do CUDA_VISIBLE_DEVICES=$i DIFIX_SHARD=$i/8 $PY $W/difix_submission_h200.py $S/HB3v4_ens/renders $S/$vn/renders ${s:+--shift $s} > $V/jobs/w22_${vn}_$i.log 2>&1 & done; wait
done
$PY $W/tta_average.py $S/HB3v4_TTA/renders $S/HB3v4_difix/renders $S/HB3v4_difix_s1/renders $S/HB3v4_difix_s2/renders --procs 90
NAME=C9_hb3v4_tp15_f101; i=0
for c in $CASES; do
  CUDA_VISIBLE_DEVICES=$i $PY $W/composite2.py tree $S/HB3v4_ens/renders $S/HB3v4_TTA/renders 0.85 $S/BGX_tp15/renders 1.0 $S/$NAME/renders --cases $c --feather 101 > $V/jobs/cand_${NAME}_$c.log 2>&1 &
  i=$((i+1))
done; wait
n=$(cnt $S/$NAME); echo "-- $NAME: $n/2056"
[ "$n" -eq 2056 ] && $PY $W/pack_submission.py $S/$NAME.zip $S/$NAME/renders && chmod 444 $S/$NAME.zip && echo "PACKED $NAME $(date +%H:%M)"
echo W22_CHAIN_DONE $(date +%H:%M)
