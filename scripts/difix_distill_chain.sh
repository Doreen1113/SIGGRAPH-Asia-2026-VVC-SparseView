#!/bin/bash
# Difix 3D-distillation member on val 001_1. Gates are LOG MARKERS (not process names).
C=001_1_seq0; R=/home/intern_2603055/vvc/runs/siga_$C; D=/home/intern_2603055/vvc/data/$C
PD=/home/intern_2603055/vvc/work/difix_pseudo_001_1
EV=/home/intern_2603055/vvc/work/eval_full.py; CU="--scale 1.0 --every 200 --cull_near_frac 1.1 --cull_min_views 4"
SIXM="run_FULLRES,run_P8,run_P9,run_J_dense01,run_GATEONLY,run_G4M"
until grep -q "TTA DONE\|FAILED" /home/intern_2603055/vvc/tta.log 2>/dev/null; do sleep 60; done
echo "== build pseudo set $(date -u +%H:%M)"
/home/intern_2603055/vvc/ftg.sh /home/intern_2603055/vvc/work/build_difix_pseudo.py $C $SIXM $PD 20 0.5 || { echo FAILED-render; exit 1; }
cd /home/intern_2603055/projects/Difix3D; export HF_HOME=/home/intern_2603055/.cache/huggingface
/home/intern_2603055/vvc/envs/difix/bin/python /home/intern_2603055/vvc/work/difix_ft_apply.py $PD/dump difix > /home/intern_2603055/vvc/difix_pseudo_apply.log 2>&1 || { echo FAILED-difix; exit 1; }
python3 - <<'PYX'
import json, shutil; from pathlib import Path
out=Path('/home/intern_2603055/vvc/work/difix_pseudo_001_1')
for e in json.load(open(out/'dump'/'manifest.json')):
    dst=out/f"{e['frame']:06d}"/f"{e['view']}.png"; dst.parent.mkdir(exist_ok=True); shutil.copy(out/'dump'/f"{e['tag']}_difix.png", dst)
print('exported', len(json.load(open(out/'index.json'))))
PYX
until grep -q "LP06 PROBE DONE\|FAILED" /home/intern_2603055/vvc/lp06_member.log 2>/dev/null; do sleep 120; done
OUT=$R/run_FULLRES_DFX
echo "== train FULLRES_DFX (S03 full-res + Difix pseudo-views, P8 recipe) $(date -u +%H:%M)"
FTGSPP_PSEUDO_W=0.15 FTGSPP_PSEUDO_LPIPS=0.03 FTGSPP_PSEUDO_START=8000 FTGSPP_PSEUDO_DIR=$PD FTGSPP_PSEUDO_COLORFIT=2 FTGSPP_PSEUDO_COLORFIT_DAMP=0.5 \
FTGSPP_LPIPS_W=0.3 FTGSPP_DEPTH_W=0.2 FTGSPP_REG_OPACITY=0.05 FTGSPP_DENSE_W=0.1 \
FTGSPP_DENSE_DEPTH_DIR=/home/intern_2603055/vvc/depth_vggt/$C FTGSPP_POINTS_DIR=$R/points_stride10 FTGSPP_TRAIN_EVAL_INTERVAL=0 FTGSPP_CHECKPOINT_INTERVAL=0 \
  /home/intern_2603055/vvc/ftg.sh -m ftgspp.run data=siga_$C run.gpu=0 data.scale=1.0 \
  train.color_correction=true train.lpips_loss=true run.output_path=$OUT > /home/intern_2603055/vvc/runs_${C}_FULLRES_DFX.log 2>&1 || { echo FAILED-train; exit 1; }
SIX="$R/run_P8,$R/run_P9,$R/run_J_dense01,$R/run_GATEONLY,$R/run_G4M"
echo "ref 6-member (FULLRES+5): 26.528/0.9348/0.2250"
echo -n "DFX solo                : "; /home/intern_2603055/vvc/ftg.sh $EV $OUT $D $CU 2>&1 | grep MEAN
echo -n "DFX replaces FULLRES (6): "; /home/intern_2603055/vvc/ftg.sh $EV $OUT $D $CU --extra_runs $SIX 2>&1 | grep MEAN
echo -n "DFX added as 7th        : "; /home/intern_2603055/vvc/ftg.sh $EV $R/run_FULLRES $D $CU --extra_runs $SIX,$OUT 2>&1 | grep MEAN
echo "DFX PROBE DONE $(date -u +%H:%M)"
