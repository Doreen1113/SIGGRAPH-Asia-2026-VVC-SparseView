#!/bin/bash
#SBATCH --job-name=val_combo3
#SBATCH --account=GOV114009
#SBATCH --partition=8gpus
#SBATCH --exclude=25a-hgpn144,25a-hgpn111
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --time=02:00:00
#SBATCH --output=/work/doreen071/vvc/jobs/val_combo3_%j.log
# Joint choice of background mix x feather x alpha, now that everything is rebuildable (peer checkpoints are back).
# All backgrounds project toward the PEER Difix TTA (ptta), lam 15 unless noted. Person = S5 Difix TTA proxy.
V=/work/doreen071/vvc; W=$V/work; D=$V/fixtest/001_1; C=$D/ps_cur; P=$D/ps_bgv3; H=$V/fixtest_h200/001_1_seq0; PY=/home/doreen071/.conda/envs/vvc/bin/python
export OMP_NUM_THREADS=10 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
$PY - "$P" "$H" "$D" <<'PYEOF'
import sys, json, numpy as np, cv2
P, H, D = sys.argv[1:4]
def rd(p): return cv2.imread(p).astype(np.float64)
T10 = ['FGW40_L60K', 'FGW40_L60K_s1', 'FGW20_s0', 'FGW80_s0', 'LONG60K_s0', 'S3_FGW40_L60K']
for e in json.load(open(P + '/manifest.json')):
    t = e['tag']; peer = rd(f'{D}/{t}_render.png'); own = rd(f'{P}/{t}_ownT.png')
    t10 = np.mean([rd(f'{H}/m_{m}/{t}_render.png') for m in T10], 0)
    for name, img in {'p2own': (2*peer + 12*own)/14, 'p4own': (4*peer + 12*own)/16, 'p6own': (6*peer + 12*own)/18,
                      'ownT2': (12*own + 4*t10)/16}.items():
        cv2.imwrite(f'{P}/{t}_{name}.png', np.clip(img, 0, 255).round().astype(np.uint8))
print('mixes built')
PYEOF
for spec in "p2own 15 bp2" "p4own 15 bp4" "p6own 15 bp6" "ownT2 15 bo2" "ownT 20 tp20" "ownT 10 tp10"; do
  set -- $spec; mix=$1; lam=$2; out=$3
  for f in $P/*_$mix.png; do t=$(basename $f _$mix.png); ln -sf $f $P/${t}_render.png; done
  for i in 0 1 2 3 4 5 6 7; do CUDA_VISIBLE_DEVICES=$i $PY $W/proj_multi.py dump $P ptta x_$out --nets alex:1,vgg:0.5 --lam $lam --steps 200 --batch 2 --shard $i/8 > $V/jobs/combo3_${out}_$i.log 2>&1 & done; wait
  echo "  $out ($mix lam$lam) $(ls $P/*_x_$out.png | wc -l)/24 $(date +%H:%M)"
done
for out in bp2 bp4 bp6 bo2 tp20 tp10; do for f in $P/*_x_$out.png; do t=$(basename $f _x_$out.png); ln -sf $f $C/${t}_bg_x_$out.png; done; done
comp() { CUDA_VISIBLE_DEVICES=$1 $PY $W/composite2.py dump $C tta $2 $3 1.0 $4 --feather $5 > $V/jobs/combo3_c_$4.log 2>&1; }
comp 0 0.85 bg_xl15 k_xl15_f81 81 & comp 1 0.85 bg_x_tp15 k_tp15_f25 25 & comp 2 0.85 bg_x_tp15 k_tp15_f81 81 & comp 3 0.85 bg_x_tp15 k_tp15_f101 101 &
comp 4 0.85 bg_x_bp2 k_bp2_f81 81 & comp 5 0.85 bg_x_bp4 k_bp4_f81 81 & comp 6 0.85 bg_x_bp6 k_bp6_f81 81 & comp 7 0.85 bg_x_bo2 k_bo2_f81 81 &
wait
comp 0 0.85 bg_x_tp20 k_tp20_f81 81 & comp 1 0.85 bg_x_tp10 k_tp10_f81 81 & comp 2 0.80 bg_x_tp15 k_tp15_f81_a80 81 & comp 3 0.90 bg_x_tp15 k_tp15_f81_a90 81 &
comp 4 0.85 bg_x_tp15 k_tp15_f131 131 & comp 5 0.85 bg_x_bp2 k_bp2_f101 101 &
wait
echo "variant                     FULL psnr/ssim/alex/vgg                FG psnr/ssim/alex/vgg"
echo "(c_xl15 = 現行配方;k_<背景>_f<羽化>[_a<alpha>])"
$PY $W/metrics_both.py $C c_xl15 k_xl15_f81 k_tp15_f25 k_tp15_f81 k_tp15_f101 k_tp15_f131 k_tp20_f81 k_tp10_f81 k_bp2_f81 k_bp2_f101 k_bp4_f81 k_bp6_f81 k_bo2_f81 k_tp15_f81_a80 k_tp15_f81_a90 2>&1 | grep -vE "Warning|warn|load_state"
echo VAL_COMBO3_DONE $(date +%H:%M)
