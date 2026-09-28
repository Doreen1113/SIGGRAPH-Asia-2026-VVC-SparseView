#!/bin/bash
#SBATCH --job-name=ens_ladder2
#SBATCH --account=GOV114009
#SBATCH --partition=dev
#SBATCH --exclude=25a-hgpn144,25a-hgpn111
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:8
#SBATCH --time=02:30:00
#SBATCH --output=/work/doreen071/vvc/jobs/ens_ladder2_%j.log
module load cuda/12.6 2>/dev/null
export PATH=/home/doreen071/.conda/envs/vvc312/bin:$PATH
export TORCH_CUDA_ARCH_LIST=9.0
export TORCH_EXTENSIONS_DIR=$(mktemp -d /tmp/tex.XXXXXX)/py312_cu121
mkdir -p $TORCH_EXTENSIONS_DIR && cp -r /work/doreen071/vvc/envcache/gsplat_prebuilt/gsplat_cuda $TORCH_EXTENSIONS_DIR/
export OMP_NUM_THREADS=6
V=/work/doreen071/vvc; S=001_1_seq0; R=$V/runs/siga_$S
dump() { CUDA_VISIBLE_DEVICES=$1 python $V/work/dump_val_renders_h200.py $S $R/run_FULLRES_s0 "$3" $V/fixtest_h200/$S/$2 --every 200 > $V/jobs/ens_$2.log 2>&1 & }
dump 0 e7 "$R/run_J_s0,$R/run_GATE_s0,$R/run_G4M_s0,$R/run_G3M_s0,$R/run_FULLRES_s1,$R/run_GATE_FR_s0"
dump 1 e8 "$R/run_J_s0,$R/run_GATE_s0,$R/run_G4M_s0,$R/run_G3M_s0,$R/run_FULLRES_s1,$R/run_GATE_FR_s0,$R/run_G4M_FR_s0"
wait
export OMP_NUM_THREADS=10
PY=/home/doreen071/.conda/envs/vvc/bin/python; W=$V/work
for n in e7 e8; do
  D=$V/fixtest_h200/$S/$n
  echo "== $n Difix chain"
  for s in "" "320,288" "640,192"; do
    case "$s" in "") vn=repo199;; "320,288") vn=repo199s1;; *) vn=repo199s2;; esac
    for i in 0 1 2 3 4 5 6 7; do CUDA_VISIBLE_DEVICES=$i $PY $W/difix_dump_h200.py $D $vn ${s:+--shift $s} --shard $i/8 > $V/jobs/e2_${n}_${vn}_$i.log 2>&1 & done
    wait
  done
  $PY - <<PYEOF
import json, numpy as np, cv2
d = "$D"
for e in json.load(open(d+'/manifest.json')):
    t = e['tag']
    ims = [cv2.imread(f'{d}/{t}_{v}.png').astype(np.float32) for v in ('repo199','repo199s1','repo199s2')]
    cv2.imwrite(f'{d}/{t}_repo199tta.png', np.clip(np.mean(ims,0),0,255).round().astype(np.uint8))
PYEOF
  CUDA_VISIBLE_DEVICES=0 $PY $W/proj_multi.py dump $D repo199tta pAVt_l15 --nets alex:1,vgg:0.5 --lam 15 --steps 200 --batch 4 > $V/jobs/e2_${n}_pAVt15.log 2>&1
  CUDA_VISIBLE_DEVICES=0 $PY $W/composite2.py dump $D repo199tta 0.50 pAVt_l15 1.0 pbt_p050_bpAVt15
  CUDA_VISIBLE_DEVICES=0 $PY $W/composite2.py dump $D repo199tta 0.60 pAVt_l15 1.0 pbt_p060_bpAVt15
  $PY $W/metrics_both.py $D render pb_p050_b045 pbt_p050_bpAVt15 pbt_p060_bpAVt15 2>&1 | grep -vE "Warning|warn|load_state"
done
echo ENS_LADDER2_DONE
