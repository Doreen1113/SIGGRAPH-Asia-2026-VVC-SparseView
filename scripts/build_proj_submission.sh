#!/bin/bash
SRC=/home/intern_2603055/vvc/submissions/SIX_ens/renders
FIX=/home/intern_2603055/vvc/submissions/SIX_difix/renders
OUT=/home/intern_2603055/vvc/submissions/SIX_PROJ_l10
cd /home/intern_2603055/vvc
echo "== projection submission build START $(date -u +%m-%d\ %H:%M) =="
./ftg.sh /home/intern_2603055/vvc/work/projection_submission.py $SRC $FIX $OUT --lam 10 --steps 200 --batch 4 2>&1 | tee /home/intern_2603055/vvc/proj_submission.log
echo "== integrity check =="
for c in $(ls $SRC); do echo "$c: $(find $OUT/$c -name '*.jpg' | wc -l) / $(find $SRC/$c -name '*.jpg' | wc -l)"; done
echo "PROJ SUBMISSION BUILD DONE $(date -u +%m-%d\ %H:%M)"
