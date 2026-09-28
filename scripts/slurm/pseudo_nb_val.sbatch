#!/bin/bash
#SBATCH --job-name=pseudo_nb_val
#SBATCH --account=GOV114009
#SBATCH --partition=8gpus
#SBATCH --exclude=25a-hgpn144,25a-hgpn111
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --time=05:00:00
#SBATCH --output=/work/doreen071/vvc/jobs/pseudo_nb_val_%j.log
module load cuda/12.6 2>/dev/null
export PATH=/home/doreen071/.conda/envs/vvc312/bin:$PATH TORCH_CUDA_ARCH_LIST=9.0 OMP_NUM_THREADS=10 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_EXTENSIONS_DIR=$(mktemp -d /tmp/tex.XXXXXX)/py312_cu121
mkdir -p $TORCH_EXTENSIONS_DIR && cp -r /work/doreen071/vvc/envcache/gsplat_prebuilt/gsplat_cuda $TORCH_EXTENSIONS_DIR/
V=/work/doreen071/vvc; W=$V/work; S=001_1_seq0; R=$V/runs/siga_$S; PY=/home/doreen071/.conda/envs/vvc/bin/python
D=$V/fixtest_h200/$S/nbteach_e20
if [ ! -f $D/manifest.json ]; then
  EX="$R/run_J_s0,$R/run_GATE_s0,$R/run_G4M_s0,$R/run_G3M_s0,$R/run_FULLRES_s1,$R/run_GATE_FR_s0"
  EX="$EX,$R/run_DFX_s0,$R/run_DFX_s0,$R/run_DFXFG_L60K,$R/run_DFXFG_L60K,$R/run_S5_FGW40_L60K,$R/run_S5_FGW40_L60K"
  CUDA_VISIBLE_DEVICES=0 python $W/dump_val_renders_h200.py $S $R/run_FULLRES_s0 "$EX" $D --every 20 > $V/jobs/pseudo_nb_val_dump.log 2>&1
fi
echo "teacher frames: $(ls $D/*_render.png 2>/dev/null | wc -l)"
for s in "" "320,288" "640,192"; do
  case "$s" in "") vn=repo199;; "320,288") vn=repo199s1;; *) vn=repo199s2;; esac
  for i in 0 1 2 3 4 5 6 7; do CUDA_VISIBLE_DEVICES=$i $PY $W/difix_dump_h200.py $D $vn ${s:+--shift $s} --shard $i/8 > /dev/null 2>&1 & done; wait
done
$PY - <<PYEOF
import json, numpy as np, cv2, os
d="$D"; tree="$D/tree/$S"
for e in json.load(open(d+'/manifest.json')):
    t=e['tag']; ims=[cv2.imread(f'{d}/{t}_{v}.png').astype(np.float32) for v in ('repo199','repo199s1','repo199s2')]
    os.makedirs(f"{tree}/{e['view']}", exist_ok=True)
    cv2.imwrite(f"{tree}/{e['view']}/{e['frame']:06d}.jpg", np.clip(np.mean(ims,0),0,255).round().astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 95])
print('tta tree written')
PYEOF
rm -rf $V/pseudo_nb/$S
/home/doreen071/.conda/envs/vvc312/bin/python $W/tta_to_pseudo.py $S $D/tree $V/pseudo_nb/$S --every 1 --scale 0.5
echo "val pseudo: $(ls $V/pseudo_nb/$S/img 2>/dev/null | wc -l)"; echo PSEUDO_NB_VAL_DONE $(date +%H:%M)
