#!/bin/bash
#SBATCH --job-name=val_personsrc
#SBATCH --account=GOV114009
#SBATCH --partition=8gpus
#SBATCH --exclude=25a-hgpn144,25a-hgpn111
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --time=03:00:00
#SBATCH --output=/work/doreen071/vvc/jobs/val_personsrc_%j.log
# Raw val FG-SSIM: S5 solo 0.8582, peer4 0.8670, diverse mix peer4+DFX2+DFXFG2+S5x2 0.8754, +S3x2 0.8788.
# Our person source is S5-dominated. Test the diverse mix AS THE PERSON SOURCE (Difix TTA, alpha 0.85 over itself),
# and 50/50 blends with S5, all over the same peer projected background, so only the person source changes.
V=/work/doreen071/vvc; W=$V/work; D=$V/fixtest/001_1; H=$V/fixtest_h200/001_1_seq0
PY=/home/doreen071/.conda/envs/vvc/bin/python
export OMP_NUM_THREADS=10 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
$PY - <<'PYEOF'
import json, numpy as np, cv2
D='/work/doreen071/vvc/fixtest/001_1'; H='/work/doreen071/vvc/fixtest_h200/001_1_seq0'
man=json.load(open(f'{D}/manifest.json'))
for e in man:
    t=e['tag']
    s5=cv2.imread(f'{H}/m_S5_FGW40_L60K/{t}_render.png').astype(np.float64)
    for mix in ('bg_p4x2d2s','bg_p4d2s3'):
        m=cv2.imread(f'{D}/{t}_{mix}.png').astype(np.float64)
        cv2.imwrite(f'{D}/{t}_{mix}_h.png', np.clip((m+s5)/2,0,255).round().astype(np.uint8))
print('blends built')
PYEOF
for SRC in bg_p4x2d2s bg_p4d2s3 bg_p4x2d2s_h bg_p4d2s3_h; do
  S=$D/ps_$SRC; mkdir -p $S; cp $D/manifest.json $S/
  for f in $D/*_gt.png $D/*_ref.png; do ln -sf $f $S/; done
  for f in $D/*_pAVt_l15.png; do t=$(basename $f _pAVt_l15.png); ln -sf $f $S/${t}_bg.png; done
  for f in $D/*_${SRC}.png; do t=$(basename $f _${SRC}.png); ln -sf $f $S/${t}_render.png; done
done
echo "== difix 3 shifts x 4 sources (8 GPUs, 2 sources at a time), $(date +%H:%M) =="
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
run_difix $D/ps_bg_p4x2d2s 0 & run_difix $D/ps_bg_p4d2s3 4 & wait
run_difix $D/ps_bg_p4x2d2s_h 0 & run_difix $D/ps_bg_p4d2s3_h 4 & wait
echo "== composites (alpha 0.85 and 0.70), $(date +%H:%M) =="
i=0
for SRC in bg_p4x2d2s bg_p4d2s3 bg_p4x2d2s_h bg_p4d2s3_h; do
  for PA in 0.85 0.70; do
    TAG=$(printf "p%03d" $(echo "$PA*100/1" | bc))
    CUDA_VISIBLE_DEVICES=$((i%8)) $PY $W/composite2.py dump $D/ps_$SRC tta $PA bg 1.0 $TAG > /dev/null 2>&1 &
    i=$((i+1))
  done
done
wait
echo "對照(S5 人物來源,d00): 26.478/0.9274/0.1478/0.3085    24.648/0.8675/0.1934/0.3246"
for SRC in bg_p4x2d2s bg_p4d2s3 bg_p4x2d2s_h bg_p4d2s3_h; do
  echo "=== person source = $SRC ==="
  $PY $W/metrics_both.py $D/ps_$SRC p085 p070 2>&1 | grep -vE "Warning|warn|load_state"
done
echo VAL_PERSONSRC_DONE $(date +%H:%M)
