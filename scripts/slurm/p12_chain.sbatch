#!/bin/bash
#SBATCH --job-name=p12_chain
#SBATCH --account=GOV114009
#SBATCH --partition=8gpus
#SBATCH --exclude=25a-hgpn144,25a-hgpn111
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --time=05:00:00
#SBATCH --output=/work/doreen071/vvc/jobs/p12_chain_%j.log
# P12 (full-res + 4M gaussians + SEVA pseudo) as a 7th peer member: SEV_ens = mean(6 peer + P12) -> NB3v2 -> HB3v2
# -> Difix x3 -> composite over BGX_tp15 at alpha 0.85 / feather 101 -> C5 package.
module load cuda/12.6 2>/dev/null
V=/work/doreen071/vvc; W=$V/work; S=$V/submissions; PY=/home/doreen071/.conda/envs/vvc/bin/python
export OMP_NUM_THREADS=10 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cnt() { find $1 -name '*.jpg' 2>/dev/null | wc -l; }
CASES="004_1_seq0 006_1_seq0 007_0_seq0 009_0_seq0 011_0_seq0"
for c in $CASES; do until [ -f $V/runs/siga_$c/run_P12_FR_G4M/gaussians.pt ] && grep -q "pipeline completed" $V/jobs/train_P12_FR_G4M_$c.log 2>/dev/null; do sleep 60; done; done
echo "== render M_P12, $(date +%H:%M) =="
i=0
for c in $CASES; do
  OUT=$S/M_P12_FR_G4M/renders/$c
  [ "$(cnt $OUT)" -gt 0 ] && { i=$((i+1)); continue; }
  ( TD=$(mktemp -d /tmp/tex.XXXXXX)/py312_cu121; mkdir -p $TD; cp -r $V/envcache/gsplat_prebuilt/gsplat_cuda $TD/
    PATH=/home/doreen071/.conda/envs/vvc312/bin:$PATH TORCH_CUDA_ARCH_LIST=9.0 TORCH_EXTENSIONS_DIR=$TD CUDA_VISIBLE_DEVICES=$i python $W/render_hidden.py $V/runs/siga_$c/run_P12_FR_G4M $V/data/$c $OUT --cull_near_frac 1.1 --cull_min_views 4 > $V/jobs/renderM_P12_$c.log 2>&1; rm -rf $(dirname $TD) ) &
  i=$((i+1))
done; wait
echo "M_P12: $(cnt $S/M_P12_FR_G4M)/2056"
echo "== SEV_ens / NB3v2 / HB3v2, $(date +%H:%M) =="
$PY $W/weighted_ensemble.py $S/SEV_ens/renders $S/SIX_ens/renders:6 $S/M_P12_FR_G4M/renders:1 --procs 90
$PY $W/weighted_ensemble.py $S/NB3v2_ens/renders $S/SEV_ens/renders:4 $S/M_DFX_s0/renders:2 $S/M_DFXFG_L60K/renders:1 $S/M_DFXFG_L60K_s1/renders:1 $S/S5P_ens/renders:2 $S/M_S3_FGW40_L60K/renders:2 --procs 90
$PY $W/tta_average.py $S/HB3v2_ens/renders $S/S5P_ens/renders $S/NB3v2_ens/renders --procs 90
echo "HB3v2_ens: $(cnt $S/HB3v2_ens)/2056"
for s in "" "320,288" "640,192"; do
  case "$s" in "") vn=HB3v2_difix;; "320,288") vn=HB3v2_difix_s1;; *) vn=HB3v2_difix_s2;; esac
  echo "== difix $vn, $(date +%H:%M) =="
  for i in 0 1 2 3 4 5 6 7; do CUDA_VISIBLE_DEVICES=$i DIFIX_SHARD=$i/8 $PY $W/difix_submission_h200.py $S/HB3v2_ens/renders $S/$vn/renders ${s:+--shift $s} > $V/jobs/p12_${vn}_$i.log 2>&1 & done; wait
done
$PY $W/tta_average.py $S/HB3v2_TTA/renders $S/HB3v2_difix/renders $S/HB3v2_difix_s1/renders $S/HB3v2_difix_s2/renders --procs 90
echo "HB3v2_TTA: $(cnt $S/HB3v2_TTA)/2056"
NAME=C5_hb3v2_tp15_f101; i=0
for c in $CASES; do
  CUDA_VISIBLE_DEVICES=$i $PY $W/composite2.py tree $S/HB3v2_ens/renders $S/HB3v2_TTA/renders 0.85 $S/BGX_tp15/renders 1.0 $S/$NAME/renders --cases $c --feather 101 > $V/jobs/cand_${NAME}_$c.log 2>&1 &
  i=$((i+1))
done; wait
n=$(cnt $S/$NAME); echo "-- $NAME: $n/2056"
[ "$n" -eq 2056 ] && $PY $W/pack_submission.py $S/$NAME.zip $S/$NAME/renders && chmod 444 $S/$NAME.zip && echo "PACKED $NAME $(date +%H:%M)"
echo P12_CHAIN_DONE $(date +%H:%M)
