#!/bin/bash
cd /home/intern_2603055/vvc; W=/home/intern_2603055/vvc/work; S=/home/intern_2603055/vvc/submissions
/home/intern_2603055/vvc/envs/difix/bin/python $W/blend_trees.py $S/S03PPJ_ens/renders $S/S03PPJ_difix_tiled/renders $S/S03PPJ_blendD/renders --map "$(cat $W/alpha_pvD.json)" --default 0.3 | tail -1
ok=1; for c in 004_1_seq0:440 006_1_seq0:416 007_0_seq0:424 009_0_seq0:416 011_0_seq0:360; do n=$(find $S/S03PPJ_blendD/renders/${c%%:*} -name '*.jpg' | wc -l); [ "$n" = "${c##*:}" ] || ok=0; done
[ $ok = 1 ] || { echo "INTEGRITY FAIL"; exit 1; }
python3 $W/mix_submission.py /home/intern_2603055/vvc/difixD_all.zip $S/S03PPJ_blendD/renders | tail -1
$W/submit_when_ready.sh /home/intern_2603055/vvc/difixD_all.zip difixD_all.zip
