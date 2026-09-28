#!/bin/bash
#SBATCH --job-name=val_nbdblend
#SBATCH --account=GOV114009
#SBATCH --partition=dev
#SBATCH --exclude=25a-hgpn144,25a-hgpn111
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --time=01:30:00
#SBATCH --output=/work/doreen071/vvc/jobs/val_nbdblend_%j.log
# Val-scene proxy for NBDB: person = mean(6x S5 seeds + FGW20/FGW80/LONG60K + NBD_S5_FG), Difix TTA, over the
# already-validated cross-target background. Goal: confirm all six move the right way (or don't) before packing.
V=/work/doreen071/vvc; W=$V/work; D=$V/fixtest/001_1; H=$V/fixtest_h200/001_1_seq0; PY=/home/doreen071/.conda/envs/vvc/bin/python
export OMP_NUM_THREADS=10 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
S=$D/ps_nbdblend; mkdir -p $S; cp $D/manifest.json $S/
for f in $D/*_gt.png $D/*_ref.png; do ln -sf $f $S/; done
$PY - <<'PYEOF'
import json, numpy as np, cv2
D='/work/doreen071/vvc/fixtest/001_1'; H='/work/doreen071/vvc/fixtest_h200/001_1_seq0'
man=json.load(open(f'{D}/manifest.json'))
parts=['m_S5_FGW40_L60K','m_FGW20_s0','m_FGW80_s0','m_LONG60K_s0','m_NBD_S5_FG']
for e in man:
    t=e['tag']; acc=None; n=0
    for p in parts:
        f=f'{H}/{p}/{t}_render.png'; im=cv2.imread(f)
        if im is None: continue
        im=im.astype(np.float64); acc=im if acc is None else acc+im; n+=1
    if acc is not None:
        cv2.imwrite(f'{D}/{t}_nbdblend.png', np.clip(acc/n,0,255).round().astype(np.uint8))
print('blend built, parts used:', n)
PYEOF
for f in $D/*_nbdblend.png; do t=$(basename $f _nbdblend.png); ln -sf $f $S/${t}_render.png; done
for f in $D/*_bg_p4x2d2s.png; do t=$(basename $f _bg_p4x2d2s.png); [ -f $S/${t}_render.png ] && ln -sf $f $S/${t}_bg.png; done
echo "staged: $(ls $S/*_render.png 2>/dev/null | wc -l)"
echo "== difix 3 shifts, $(date +%H:%M) =="
for s in "" "320,288" "640,192"; do
  case "$s" in "") vn=repo199;; "320,288") vn=repo199s1;; *) vn=repo199s2;; esac
  for i in 0 1 2 3 4 5 6 7; do CUDA_VISIBLE_DEVICES=$i $PY $W/difix_dump_h200.py $S $vn ${s:+--shift $s} --shard $i/8 > /dev/null 2>&1 & done; wait
done
$PY - <<'PYEOF'
import json, numpy as np, cv2
d='/work/doreen071/vvc/fixtest/001_1/ps_nbdblend'
for e in json.load(open(d+'/manifest.json')):
    t=e['tag']; ims=[cv2.imread(f'{d}/{t}_{v}.png') for v in ('repo199','repo199s1','repo199s2')]
    if any(i is None for i in ims): continue
    ims=[i.astype(np.float32) for i in ims]
    cv2.imwrite(f'{d}/{t}_tta.png', np.clip(np.mean(ims,0),0,255).round().astype(np.uint8))
PYEOF
i=0
for PA in 0.85 0.70; do
  TAG=$(printf "p%03d" $(echo "$PA*100/1" | bc))
  CUDA_VISIBLE_DEVICES=$((i%8)) $PY $W/composite2.py dump $S tta $PA bg 1.0 $TAG > /dev/null 2>&1 &
  i=$((i+1))
done
wait
echo "variant                     FULL psnr/ssim/alex/vgg                FG psnr/ssim/alex/vgg"
echo "(對照:純stride5+此背景 j_p070_xl15: 27.082/0.9297/0.1467 | 25.029/0.8737/0.1888)"
echo "(對照:現行送出 HB3系列 p085 大約: 27.0~/0.921~/0.181~ | 25.0~/0.841~/0.234~)"
$PY $W/metrics_both.py $S p085 p070 2>&1 | grep -vE "Warning|warn|load_state"
echo VAL_NBDBLEND_DONE $(date +%H:%M)
