#!/bin/bash
#SBATCH --job-name=val_dilate
#SBATCH --account=GOV114009
#SBATCH --partition=8gpus
#SBATCH --exclude=25a-hgpn144,25a-hgpn111
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --time=03:00:00
#SBATCH --output=/work/doreen071/vvc/jobs/val_dilate_%j.log
# Our DeepLab person mask covers 0.0526 of the frame; the evaluator's foreground is 0.0596, so ~12% of the
# region it scores is filled from the background instead of the person source. When the person source was WORSE
# than the peer on FG-SSIM that was protective; S6E's person source now BEATS it (0.83496 vs peer 0.83203), so
# widening the mask should now push better content into the scored ring. FG-SSIM is our cheapest rank ladder
# (+0.0073 = three ranks = FINAL 2.333), so measure the dilation sweep on val.
V=/work/doreen071/vvc; W=$V/work; D=$V/fixtest/001_1; H=$V/fixtest_h200/001_1_seq0
PY=/home/doreen071/.conda/envs/vvc/bin/python
export OMP_NUM_THREADS=10 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
S=$D/dil_s5; mkdir -p $S; cp $D/manifest.json $S/
# raw = the S5 person-source render; keep the peer's projected background as the B variant
for f in $D/*_gt.png $D/*_ref.png; do ln -sf $f $S/; done
for f in $D/*_pAVt_l15.png; do t=$(basename $f _pAVt_l15.png); ln -sf $f $S/${t}_bg.png; done
n=0
for f in $H/m_S5_FGW40_L60K/*_render.png; do t=$(basename $f _render.png); ln -sf $f $S/${t}_render.png; n=$((n+1)); done
echo "staged $n S5 renders; bg variants: $(ls $S/*_bg.png | wc -l)"
echo "== difix 3 shifts on the S5 person source, $(date +%H:%M) =="
for s in "" "320,288" "640,192"; do
  case "$s" in "") vn=repo199;; "320,288") vn=repo199s1;; *) vn=repo199s2;; esac
  for i in 0 1 2 3 4 5 6 7; do CUDA_VISIBLE_DEVICES=$i $PY $W/difix_dump_h200.py $S $vn ${s:+--shift $s} --shard $i/8 > /dev/null 2>&1 & done; wait
done
$PY - <<PYEOF
import json, numpy as np, cv2
d="$S"
for e in json.load(open(d+'/manifest.json')):
    t=e['tag']; ims=[cv2.imread(f'{d}/{t}_{v}.png').astype(np.float32) for v in ('repo199','repo199s1','repo199s2')]
    cv2.imwrite(f'{d}/{t}_tta.png', np.clip(np.mean(ims,0),0,255).round().astype(np.uint8))
PYEOF
echo "== dilation sweep (alpha 0.85), $(date +%H:%M) =="
i=0
for DIL in 0 8 16 24 32 48; do
  TAG=$(printf "d%02d" $DIL)
  CUDA_VISIBLE_DEVICES=$((i%8)) $PY $W/composite2.py dump $S tta 0.85 bg 1.0 $TAG --dilate $DIL > $V/jobs/valdil_$TAG.log 2>&1 &
  i=$((i+1))
done
wait
echo "variant                     FULL psnr/ssim/alex/vgg                FG psnr/ssim/alex/vgg"
$PY $W/metrics_both.py $S d00 d08 d16 d24 d32 d48 2>&1 | grep -vE "Warning|warn|load_state"
echo VAL_DILATE_DONE $(date +%H:%M)
