#!/bin/bash
# after the weak-scene tiled pass finishes, run the remaining three scenes so a full Difix-blend submission exists
export HF_HOME=/home/intern_2603055/.cache/huggingface
while ps -eo args | grep -v grep | grep -q "difix_submission.py.*011_0_seq0,007_0_seq0"; do sleep 120; done
/home/intern_2603055/vvc/envs/difix/bin/python /home/intern_2603055/vvc/work/difix_submission.py /home/intern_2603055/vvc/submissions/S03PPJ_ens/renders /home/intern_2603055/vvc/submissions/S03PPJ_difix_tiled/renders --mode tiled --cases 004_1_seq0,006_1_seq0,009_0_seq0
echo "ALL SCENES DIFIX DONE $(date -u +%H:%M)"
