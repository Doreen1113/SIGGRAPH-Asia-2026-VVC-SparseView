#!/bin/bash
#SBATCH --job-name=val_swapft
#SBATCH --account=GOV114009
#SBATCH --partition=dev
#SBATCH --exclude=25a-hgpn144,25a-hgpn111
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:4
#SBATCH --time=00:40:00
#SBATCH --output=/work/doreen071/vvc/jobs/val_swapfeather_%j.log
# Can the feather change be applied to a PACKED composite (person tree deleted)? Simulate on val:
#   c_xl15 (k25, JPEG-roundtripped like the zip) -> swap_feather k25->k81 / k61 with fallback F = B or F = another
#   model's raw (DFX_s0, pessimistic stand-in for the S5 seed mean available on test). Compare to the true f81/f61.
V=/work/doreen071/vvc; W=$V/work; C=$V/fixtest/001_1/ps_cur; H=$V/fixtest_h200/001_1_seq0; PY=/home/doreen071/.conda/envs/vvc/bin/python
$PY - "$C" "$H" <<'PYEOF'
import sys, json, cv2
C, H = sys.argv[1:3]
for e in json.load(open(C+'/manifest.json')):
    g = e['tag']
    ok, buf = cv2.imencode('.jpg', cv2.imread(f'{C}/{g}_c_xl15.png'), [cv2.IMWRITE_JPEG_QUALITY, 95]); cv2.imwrite(f'{C}/{g}_cj.png', cv2.imdecode(buf, 1))
    cv2.imwrite(f'{C}/{g}_fbdfx.png', cv2.imread(f'{H}/m_DFX_s0/{g}_render.png'))
print('staged')
PYEOF
CUDA_VISIBLE_DEVICES=0 $PY $W/swap_feather.py dump $C cj bg_xl15 B sfB81 --k1 81 &
CUDA_VISIBLE_DEVICES=1 $PY $W/swap_feather.py dump $C cj bg_xl15 fbdfx sfD81 --k1 81 &
CUDA_VISIBLE_DEVICES=2 $PY $W/swap_feather.py dump $C cj bg_xl15 B sfB61 --k1 61 &
CUDA_VISIBLE_DEVICES=3 $PY $W/swap_feather.py dump $C cj bg_xl15 fbdfx sfD61 --k1 61 &
wait
echo "variant                     FULL psnr/ssim/alex/vgg                FG psnr/ssim/alex/vgg"
echo "(cj = 現行 k25 經 JPEG;f61/f81 = 真正重建;sfB/sfD = 從打包圖反推,fallback=背景/其他模型)"
$PY $W/metrics_both.py $C cj f61 sfB61 sfD61 f81 sfB81 sfD81 2>&1 | grep -vE "Warning|warn|load_state"
echo VAL_SWAPFT_DONE $(date +%H:%M)
