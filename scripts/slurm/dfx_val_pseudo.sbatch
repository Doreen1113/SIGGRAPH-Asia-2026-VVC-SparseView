#!/bin/bash
#SBATCH --job-name=dfx_valp
#SBATCH --account=GOV114009
#SBATCH --partition=dev
#SBATCH --exclude=25a-hgpn144,25a-hgpn111
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --time=03:55:00
#SBATCH --output=/work/doreen071/vvc/jobs/dfx_valp_%j.log
# Build Difix-TTA pseudo views for val 001_1 (the test-scene pseudo set came from NINE_TTA, which has no val scene),
# then train DFX_s0 on val so eval_dfx can measure it before we spend a submission slot on DFX.
module load cuda/12.6 2>/dev/null
export PATH=/home/doreen071/.conda/envs/vvc312/bin:$PATH TORCH_CUDA_ARCH_LIST=9.0 HYDRA_FULL_ERROR=1 OMP_NUM_THREADS=10
export TORCH_EXTENSIONS_DIR=$(mktemp -d /tmp/tex.XXXXXX)/py312_cu121
mkdir -p $TORCH_EXTENSIONS_DIR && cp -r /work/doreen071/vvc/envcache/gsplat_prebuilt/gsplat_cuda $TORCH_EXTENSIONS_DIR/
V=/work/doreen071/vvc; W=$V/work; S=001_1_seq0; R=$V/runs/siga_$S; PY=/home/doreen071/.conda/envs/vvc/bin/python
D=$V/fixtest_h200/$S/e7_e20
if [ ! -f $D/manifest.json ]; then
  echo "== dump e7 ensemble at val test poses every 20 frames, $(date -u +%H:%M) =="
  python $W/dump_val_renders_h200.py $S $R/run_FULLRES_s0 "$R/run_J_s0,$R/run_GATE_s0,$R/run_G4M_s0,$R/run_G3M_s0,$R/run_FULLRES_s1,$R/run_GATE_FR_s0" $D --every 20 > $V/jobs/dfxvalp_dump.log 2>&1
fi
echo "dumped $(ls $D/*_render.png | wc -l) frames"
echo "== difix 3 shifts, $(date -u +%H:%M) =="
for s in "" "320,288" "640,192"; do
  case "$s" in "") vn=repo199;; "320,288") vn=repo199s1;; *) vn=repo199s2;; esac
  for i in 0 1 2 3 4 5 6 7; do CUDA_VISIBLE_DEVICES=$i $PY $W/difix_dump_h200.py $D $vn ${s:+--shift $s} --shard $i/8 > $V/jobs/dfxvalp_${vn}_$i.log 2>&1 & done
  wait
done
$PY - <<PYEOF
import json, numpy as np, cv2, os
d="$D"; tree="$D/tree/$S"
for e in json.load(open(d+'/manifest.json')):
    t=e['tag']
    ims=[cv2.imread(f'{d}/{t}_{v}.png').astype(np.float32) for v in ('repo199','repo199s1','repo199s2')]
    os.makedirs(f"{tree}/{e['view']}", exist_ok=True)
    cv2.imwrite(f"{tree}/{e['view']}/{e['frame']:06d}.jpg", np.clip(np.mean(ims,0),0,255).round().astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 95])
print('tta tree written')
PYEOF
rm -rf $V/pseudo/$S
python $W/tta_to_pseudo.py $S $D/tree $V/pseudo/$S --every 1 --scale 0.5
echo "== train DFX_s0 on val, $(date -u +%H:%M) =="
rm -rf $R/run_DFX_s0
export FTGSPP_RUNS_ROOT=$V/runs FTGSPP_LPIPS_W=0.3 FTGSPP_DEPTH_W=0.2 FTGSPP_REG_OPACITY=0.05 FTGSPP_DENSE_W=0.1
export FTGSPP_TRAIN_EVAL_INTERVAL=0 FTGSPP_CHECKPOINT_INTERVAL=0
export FTGSPP_DENSE_DEPTH_DIR=$V/depth_vggt/$S FTGSPP_POINTS_DIR=$R/points_stride10
export FTGSPP_PSEUDO_DIR=$V/pseudo/$S FTGSPP_PSEUDO_W=0.5 FTGSPP_PSEUDO_LPIPS=0.1 FTGSPP_PSEUDO_START=3000
export FTGSPP_PSEUDO_MASK=0 FTGSPP_PSEUDO_COVERAGE=0 FTGSPP_PSEUDO_AGREE=0
cd $V/repo/baseline_code
CUDA_VISIBLE_DEVICES=0 python -m ftgspp.run data=siga_$S run.gpu=0 \
  data.root=$V/data/$S data.video_path=$V/data/$S data.calibration_path=$V/data/$S data.extracted_path=$V/data/$S/images \
  train.color_correction=true train.lpips_loss=true train.num_workers=16 train.prefetch_factor=4 data.scale=1.0 \
  run.output_path=$R/run_DFX_s0 > $V/jobs/train_DFX_s0_$S.log 2>&1
[ -f $R/run_DFX_s0/gaussians.pt ] && echo "TRAIN_DFX_val_DONE $(date -u +%H:%M)" || echo "TRAIN_DFX_val_FAILED"
