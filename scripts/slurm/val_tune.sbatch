#!/bin/bash
#SBATCH --job-name=val_tune
#SBATCH --account=GOV114009
#SBATCH --partition=8gpus
#SBATCH --exclude=25a-hgpn144,25a-hgpn111
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --time=02:00:00
#SBATCH --output=/work/doreen071/vvc/jobs/val_tune_%j.log
# Estimated test FINAL 2.500 for HB3 p070 + cross background. Next rungs: FG-SSIM +0.0011, FG-PSNR +0.153.
# Tune on val with the xl15 background: (a) lower person alpha on the current half blend; (b) mix fraction
# 35% / 65% in the person source (half = 50% was +0.0046 FG-SSIM, pure mix was negative).
V=/work/doreen071/vvc; W=$V/work; D=$V/fixtest/001_1; H=$V/fixtest_h200/001_1_seq0; PY=/home/doreen071/.conda/envs/vvc/bin/python
export OMP_NUM_THREADS=10 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=$D/ps_bg_p4x2d2s
$PY - <<'PYEOF'
import json, numpy as np, cv2
D='/work/doreen071/vvc/fixtest/001_1'; H='/work/doreen071/vvc/fixtest_h200/001_1_seq0'
for e in json.load(open(f'{D}/manifest.json')):
    t=e['tag']; s5=cv2.imread(f'{H}/m_S5_FGW40_L60K/{t}_render.png').astype(np.float64); m=cv2.imread(f'{D}/{t}_bg_p4d2s3.png').astype(np.float64)
    for name,w in (('m35',0.35),('m65',0.65)):
        cv2.imwrite(f'{D}/{t}_bg_p4d2s3_{name}.png', np.clip((1-w)*s5+w*m,0,255).round().astype(np.uint8))
print('blends built')
PYEOF
for SRC in bg_p4d2s3_m35 bg_p4d2s3_m65; do
  S=$D/ps_$SRC; mkdir -p $S; cp $D/manifest.json $S/
  for f in $D/*_gt.png $D/*_ref.png; do ln -sf $f $S/; done
  for f in $D/*_${SRC}.png; do t=$(basename $f _${SRC}.png); ln -sf $f $S/${t}_render.png; done
  for f in $P/*_xl15.png; do t=$(basename $f _xl15.png); ln -sf $f $S/${t}_bg_xl15.png; done
done
run_difix() {
  local S=$1; local g0=$2
  for s in "" "320,288" "640,192"; do
    case "$s" in "") vn=repo199;; "320,288") vn=repo199s1;; *) vn=repo199s2;; esac
    for i in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$((g0+i)) $PY $W/difix_dump_h200.py $S $vn ${s:+--shift $s} --shard $i/4 > /dev/null 2>&1 & done; wait
  done
  $PY - "$S" <<'PYEOF'
import json, sys, numpy as np, cv2
d=sys.argv[1]
for e in json.load(open(d+'/manifest.json')):
    t=e['tag']; ims=[cv2.imread(f'{d}/{t}_{v}.png').astype(np.float32) for v in ('repo199','repo199s1','repo199s2')]
    cv2.imwrite(f'{d}/{t}_tta.png', np.clip(np.mean(ims,0),0,255).round().astype(np.uint8))
PYEOF
}
run_difix $D/ps_bg_p4d2s3_m35 0 & run_difix $D/ps_bg_p4d2s3_m65 4 & wait
S50=$D/ps_bg_p4d2s3_h
g=0; OUTS=""
for spec in "$S50 0.65 h50" "$S50 0.60 h50" "$S50 0.55 h50" "$D/ps_bg_p4d2s3_m35 0.70 m35" "$D/ps_bg_p4d2s3_m35 0.60 m35" "$D/ps_bg_p4d2s3_m65 0.70 m65" "$D/ps_bg_p4d2s3_m65 0.60 m65"; do
  set -- $spec; TAG=$(printf "t%03d" $(echo "$2*100/1" | bc))_$3
  CUDA_VISIBLE_DEVICES=$((g%8)) $PY $W/composite2.py dump $1 tta $2 bg_xl15 1.0 $TAG > /dev/null 2>&1 &
  g=$((g+1))
done
wait
echo "variant                     FULL psnr/ssim/alex/vgg                FG psnr/ssim/alex/vgg"
echo "(d00 現行: 26.478/0.9274/0.1478/0.3085    24.648/0.8675/0.1934/0.3246)"
echo "(j_p070_xl15 預定送出: 27.082/0.9297/0.1467/0.3063    25.029/0.8737/0.1888/0.3123)"
echo "--- half blend, lower alpha ---"
$PY $W/metrics_both.py $S50 t065_h50 t060_h50 t055_h50 2>&1 | grep -vE "Warning|warn|load_state|^variant"
echo "--- mix 35% ---"
$PY $W/metrics_both.py $D/ps_bg_p4d2s3_m35 t070_m35 t060_m35 2>&1 | grep -vE "Warning|warn|load_state|^variant"
echo "--- mix 65% ---"
$PY $W/metrics_both.py $D/ps_bg_p4d2s3_m65 t070_m65 t060_m65 2>&1 | grep -vE "Warning|warn|load_state|^variant"
echo VAL_TUNE_DONE $(date +%H:%M)
