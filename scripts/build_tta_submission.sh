#!/bin/bash
# Build and submit the TTA-Difix candidate: six-member ensemble base, 3-shift TTA Difix, person=0.60/bg=0.55
# composite (validated on both val scenes: same SSIM as the safe pb_p50_b45 recipe, LPIPS lower by ~0.007).
set -e
SRC=/home/intern_2603055/vvc/submissions/SIX_ens/renders
DFX=/home/intern_2603055/vvc/submissions/SIX_TTA_difix
OUT=/home/intern_2603055/vvc/submissions/SIX_TTA_p60_b55
cd /home/intern_2603055/projects/Difix3D; export HF_HOME=/home/intern_2603055/.cache/huggingface
echo "== TTA Difix pass over 5 scenes, $(date -u +%H:%M) =="
/home/intern_2603055/vvc/envs/difix/bin/python /home/intern_2603055/vvc/work/difix_submission_tta.py $SRC $DFX 2>&1 | tee /home/intern_2603055/vvc/tta_submission_difix.log
echo "== person/bg composite pa=0.60 ba=0.55, $(date -u +%H:%M) =="
/home/intern_2603055/vvc/ftg.sh /home/intern_2603055/vvc/work/person_bg_composite.py $SRC $DFX $OUT --pa 0.60 --ba 0.55 2>&1 | tee /home/intern_2603055/vvc/tta_submission_composite.log
echo "== integrity check =="
for c in $(ls $SRC); do echo "$c: $(find $OUT/$c -name '*.jpg' | wc -l) / $(find $SRC/$c -name '*.jpg' | wc -l)"; done
echo "TTA SUBMISSION BUILD DONE $(date -u +%H:%M)"
