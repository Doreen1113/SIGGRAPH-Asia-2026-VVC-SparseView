#!/bin/bash
#SBATCH --job-name=val_geoquery
#SBATCH --account=GOV114009
#SBATCH --partition=8gpus
#SBATCH --exclude=25a-hgpn144,25a-hgpn111
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --time=03:00:00
#SBATCH --output=/work/doreen071/vvc/jobs/val_geoquery_%j.log
# GeoQuery vs Difix (nvidia/difix_ref) on two val scenes: 001_1 (007_0 rig) and 012_0 (011_0 rig).
# Identical pipeline otherwise: same person render, same nearest-train-camera reference, 3-shift TTA, alpha 0.85.
module load cuda/12.6 2>/dev/null
V=/work/doreen071/vvc; W=$V/work; PY=/home/doreen071/.conda/envs/vvc/bin/python
export OMP_NUM_THREADS=10 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 HF_HOME=$V/hf_home
CK=$V/ext/ckpt/geoquery_window_size3_down1.pkl
until [ -f $CK ] && ! ls $V/ext/ckpt/.cache/huggingface/download/*.incomplete >/dev/null 2>&1; do sleep 30; done
ls -la $CK
C1=$V/fixtest/001_1/ps_cur; C2=$V/fixtest/012_0/ps_lev
for f in $V/fixtest/012_0/*_repo199.png; do t=$(basename $f _repo199.png); ln -sf $f $C2/${t}_d1.png; done
for f in $C1/*_repo199.png; do t=$(basename $f _repo199.png); ln -sf $f $C1/${t}_d1.png; done
echo "== smoke test, $(date +%H:%M) =="
mkdir -p $C1/smoke && head -c 0 /dev/null
CUDA_VISIBLE_DEVICES=0 $PY $W/geoquery_dump.py $C1 gq --case $V/data/001_1_seq0 --depth $V/depth_vggt/001_1_seq0 --shard 0/24 2>&1 | grep -vE "Warning|warn" | tail -15
[ -f $C1/02_000000_gq.png ] || { echo SMOKE_FAILED; exit 1; }
run_scene() {  # dir case bgvar
  local D=$1 CASE=$2 BG=$3
  for s in "" "320,288" "640,192"; do
    case "$s" in "") vn=gq;; "320,288") vn=gq_s1;; *) vn=gq_s2;; esac
    for i in 0 1 2 3 4 5 6 7; do
      CUDA_VISIBLE_DEVICES=$i $PY $W/geoquery_dump.py $D $vn --case $V/data/$CASE --depth $V/depth_vggt/$CASE ${s:+--shift $s} --shard $i/8 > $V/jobs/gq_${CASE}_${vn}_$i.log 2>&1 &
    done; wait; echo "  $CASE $vn done $(date +%H:%M)"
  done
  for i in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES=$i $PY $W/geoquery_dump.py $D gqng --nogeo --case $V/data/$CASE --depth $V/depth_vggt/$CASE --shard $i/8 > $V/jobs/gq_${CASE}_ng_$i.log 2>&1 &
  done; wait
  $PY - "$D" <<'PYEOF'
import sys, json, numpy as np, cv2
D = sys.argv[1]
for e in json.load(open(D + '/manifest.json')):
    t = e['tag']; ims = [cv2.imread(f'{D}/{t}_{v}.png') for v in ('gq', 'gq_s1', 'gq_s2')]
    if any(i is None for i in ims): print('missing', t); continue
    cv2.imwrite(f'{D}/{t}_gqtta.png', np.clip(np.mean([i.astype(np.float32) for i in ims], 0), 0, 255).round().astype(np.uint8))
print('tta built')
PYEOF
  g=0
  for v in gq gqtta gqng d1; do
    CUDA_VISIBLE_DEVICES=$((g%8)) $PY $W/composite2.py dump $D $v 0.85 $BG 1.0 c_$v > $V/jobs/gqcomp_${CASE}_$v.log 2>&1 &
    g=$((g+1))
  done; wait
}
echo "== 001_1, $(date +%H:%M) =="; run_scene $C1 001_1_seq0 bg_xl15
echo "== 012_0, $(date +%H:%M) =="; run_scene $C2 012_0_seq0 bg
echo "variant                     FULL psnr/ssim/alex/vgg                FG psnr/ssim/alex/vgg"
echo "--- 001_1 (c_xl15 = Difix 3-shift 現行;c_d1 = Difix 單次;c_gq = GeoQuery 單次;c_gqtta = GeoQuery 3-shift;c_gqng = 關掉幾何)"
$PY $W/metrics_both.py $C1 c_xl15 c_d1 c_gq c_gqtta c_gqng 2>&1 | grep -vE "Warning|warn|load_state"
echo "--- 012_0 (f25 = Difix 3-shift 現行)"
$PY $W/metrics_both.py $C2 f25 c_d1 c_gq c_gqtta c_gqng 2>&1 | grep -vE "Warning|warn|load_state"
echo VAL_GEOQUERY_DONE $(date +%H:%M)
