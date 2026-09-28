#!/bin/bash
#SBATCH --job-name=val012_levers
#SBATCH --account=GOV114009
#SBATCH --partition=dev
#SBATCH --exclude=25a-hgpn144,25a-hgpn111
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --time=01:30:00
#SBATCH --output=/work/doreen071/vvc/jobs/val012_levers_%j.log
# 011_0 is our worst test scene; val 012_0 uses the identical camera rig (legal proxy: validation data used for
# tuning only, nothing from it enters a submission). Test the post-processing levers on this rig:
#   feather width, mask dilation, person alpha, and the zip-deployable swap_feather variants.
# person = repo199tta (peer-ensemble Difix 3-shift), background = pAVt_l15.
V=/work/doreen071/vvc; W=$V/work; D=$V/fixtest/012_0; PY=/home/doreen071/.conda/envs/vvc/bin/python
export OMP_NUM_THREADS=10 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
S=$D/ps_lev; mkdir -p $S; cp $D/manifest.json $S/
for f in $D/*_gt.png $D/*_ref.png $D/*_render.png; do ln -sf $f $S/; done
for f in $D/*_repo199tta.png; do t=$(basename $f _repo199tta.png); ln -sf $f $S/${t}_tta.png; done
for f in $D/*_pAVt_l15.png; do t=$(basename $f _pAVt_l15.png); ln -sf $f $S/${t}_bg.png; done
echo "staged gt=$(ls $S/*_gt.png|wc -l) tta=$(ls $S/*_tta.png|wc -l) bg=$(ls $S/*_bg.png|wc -l)"
run() { CUDA_VISIBLE_DEVICES=$1 $PY $W/composite2.py dump $S tta $2 bg 1.0 $3 ${@:4} > $V/jobs/v012_$3.log 2>&1; }
run 0 0.85 f25 --feather 25 & run 1 0.85 f41 --feather 41 & run 2 0.85 f61 --feather 61 & run 3 0.85 f81 --feather 81 &
run 4 0.85 f101 --feather 101 & run 5 0.85 d16 --dilate 16 & run 6 0.85 d32 --dilate 32 & run 7 0.70 a70 --feather 25 &
wait
run 0 1.00 a100 --feather 25 & run 1 0.85 f81d16 --feather 81 --dilate 16 & run 2 0.85 f131 --feather 131 &
wait
# zip-deployable approximation on this rig, from a JPEG round-trip of f25
$PY - "$S" <<'PYEOF'
import sys, json, cv2
S = sys.argv[1]
for e in json.load(open(S+'/manifest.json')):
    g = e['tag']; ok, buf = cv2.imencode('.jpg', cv2.imread(f'{S}/{g}_f25.png'), [cv2.IMWRITE_JPEG_QUALITY, 95]); cv2.imwrite(f'{S}/{g}_cj.png', cv2.imdecode(buf, 1))
PYEOF
g=0
for spec in "101 4 0.15 0.5 s4k101" "81 0 0.15 0.5 raw81" "131 4 0.15 0.5 s4k131" "101 2 0.15 0.5 s2k101"; do
  set -- $spec
  CUDA_VISIBLE_DEVICES=$g $PY $W/swap_feather.py dump $S cj bg B sf_$5 --k1 $1 --sigma $2 --t0 $3 --t1 $4 > $V/jobs/v012_sf_$5.log 2>&1 &
  g=$((g+1))
done; wait
echo "variant                     FULL psnr/ssim/alex/vgg                FG psnr/ssim/alex/vgg"
echo "(f25 = 現行合成;cj = f25 經 JPEG = 打包狀態;sf_* = 可從 zip 部署的近似)"
$PY $W/metrics_both.py $S f25 f41 f61 f81 f101 f131 d16 d32 f81d16 a70 a100 cj sf_s4k101 sf_raw81 sf_s4k131 sf_s2k101 2>&1 | grep -vE "Warning|warn|load_state"
echo VAL012_LEVERS_DONE $(date +%H:%M)
