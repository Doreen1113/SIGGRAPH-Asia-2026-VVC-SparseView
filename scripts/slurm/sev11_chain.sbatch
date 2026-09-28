#!/bin/bash
#SBATCH --job-name=sev11_chain
#SBATCH --account=GOV114009
#SBATCH --partition=8gpus
#SBATCH --exclude=25a-hgpn144,25a-hgpn111
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --time=06:00:00
#SBATCH --output=/work/doreen071/vvc/jobs/sev11_chain_%j.log
# 11-member peer ensemble: SIX_ens(6) + P12 + wave19 (P8_s1, P9_s1, P13_GATE, P14_FR), equal weight
# -> NB3v3 -> HB3v3 -> Difix x3 -> composite over BGX_tp15 alpha 0.85 feather 101 -> C7 package.
module load cuda/12.6 2>/dev/null
V=/work/doreen071/vvc; W=$V/work; S=$V/submissions; PY=/home/doreen071/.conda/envs/vvc/bin/python
export OMP_NUM_THREADS=10 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cnt() { find $1 -name '*.jpg' 2>/dev/null | wc -l; }
CASES="004_1_seq0 006_1_seq0 007_0_seq0 009_0_seq0 011_0_seq0"
render() { # $1 model $2 case $3 gpu
  OUT=$S/M_$1/renders/$2
  [ "$(cnt $OUT)" -gt 0 ] && return
  TD=$(mktemp -d /tmp/tex.XXXXXX)/py312_cu121; mkdir -p $TD; cp -r $V/envcache/gsplat_prebuilt/gsplat_cuda $TD/
  PATH=/home/doreen071/.conda/envs/vvc312/bin:$PATH TORCH_CUDA_ARCH_LIST=9.0 TORCH_EXTENSIONS_DIR=$TD CUDA_VISIBLE_DEVICES=$3 python $W/render_hidden.py $V/runs/siga_$2/run_$1 $V/data/$2 $OUT --cull_near_frac 1.1 --cull_min_views 4 > $V/jobs/renderM_$1_$2.log 2>&1
  rm -rf $(dirname $TD)
}
echo "== render wave19, $(date +%H:%M) =="
JOBS=(); for m in P8_s1 P9_s1 P13_GATE P14_FR; do for c in $CASES; do JOBS+=("$m $c"); done; done
k=0
for j in "${JOBS[@]}"; do
  render $j $((k % 8)) &
  k=$((k+1)); [ $((k % 8)) -eq 0 ] && wait
done; wait
for m in P8_s1 P9_s1 P13_GATE P14_FR; do echo "M_$m: $(cnt $S/M_$m)/2056"; done
for m in P8_s1 P9_s1 P13_GATE P14_FR; do [ "$(cnt $S/M_$m)" -eq 2056 ] || { echo "RENDER INCOMPLETE $m"; exit 1; }; done
echo "== SEV11 / NB3v3 / HB3v3, $(date +%H:%M) =="
$PY $W/weighted_ensemble.py $S/SEV11_ens/renders $S/SIX_ens/renders:6 $S/M_P12_FR_G4M/renders:1 $S/M_P8_s1/renders:1 $S/M_P9_s1/renders:1 $S/M_P13_GATE/renders:1 $S/M_P14_FR/renders:1 --procs 90
$PY $W/weighted_ensemble.py $S/NB3v3_ens/renders $S/SEV11_ens/renders:4 $S/M_DFX_s0/renders:2 $S/M_DFXFG_L60K/renders:1 $S/M_DFXFG_L60K_s1/renders:1 $S/S5P_ens/renders:2 $S/M_S3_FGW40_L60K/renders:2 --procs 90
$PY $W/tta_average.py $S/HB3v3_ens/renders $S/S5P_ens/renders $S/NB3v3_ens/renders --procs 90
echo "HB3v3_ens: $(cnt $S/HB3v3_ens)/2056"
for s in "" "320,288" "640,192"; do
  case "$s" in "") vn=HB3v3_difix;; "320,288") vn=HB3v3_difix_s1;; *) vn=HB3v3_difix_s2;; esac
  echo "== difix $vn, $(date +%H:%M) =="
  for i in 0 1 2 3 4 5 6 7; do CUDA_VISIBLE_DEVICES=$i DIFIX_SHARD=$i/8 $PY $W/difix_submission_h200.py $S/HB3v3_ens/renders $S/$vn/renders ${s:+--shift $s} > $V/jobs/sev11_${vn}_$i.log 2>&1 & done; wait
done
$PY $W/tta_average.py $S/HB3v3_TTA/renders $S/HB3v3_difix/renders $S/HB3v3_difix_s1/renders $S/HB3v3_difix_s2/renders --procs 90
echo "HB3v3_TTA: $(cnt $S/HB3v3_TTA)/2056"
NAME=C7_hb3v3_tp15_f101; i=0
for c in $CASES; do
  CUDA_VISIBLE_DEVICES=$i $PY $W/composite2.py tree $S/HB3v3_ens/renders $S/HB3v3_TTA/renders 0.85 $S/BGX_tp15/renders 1.0 $S/$NAME/renders --cases $c --feather 101 > $V/jobs/cand_${NAME}_$c.log 2>&1 &
  i=$((i+1))
done; wait
n=$(cnt $S/$NAME); echo "-- $NAME: $n/2056"
[ "$n" -eq 2056 ] && $PY $W/pack_submission.py $S/$NAME.zip $S/$NAME/renders && chmod 444 $S/$NAME.zip && echo "PACKED $NAME $(date +%H:%M)"
echo SEV11_CHAIN_DONE $(date +%H:%M)
