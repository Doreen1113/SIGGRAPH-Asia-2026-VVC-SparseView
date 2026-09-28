#!/bin/bash
# Build ensemble renders for the hidden views as soon as checkpoints land — NO uploading.
# Keeps the best package ready so submission is instant once the organizers' server works.
W=/home/intern_2603055/vvc/work; S=/home/intern_2603055/vvc/submissions; RN=/home/intern_2603055/vvc/runs
while true; do
  for c in 004_1_seq0 006_1_seq0 007_0_seq0 009_0_seq0 011_0_seq0; do
    R=$RN/siga_$c
    if [ -f $R/run_S015_s0/gaussians.pt ] && [ -f $R/run_P8_s0/gaussians.pt ] && [ -f $R/run_P9_s0/gaussians.pt ] && [ ! -d $S/S015PP_ens/renders/$c ]; then
      while [ $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) -gt 36000 ]; do sleep 60; done
      echo "== render S015PP $c $(date -u +%m-%d\ %H:%M)"
      /home/intern_2603055/vvc/ftg.sh $W/render_hidden.py $R/run_S015_s0 /home/intern_2603055/vvc/data/$c $S/S015PP_ens/renders/$c --cull_near_frac 1.1 --cull_min_views 4 --extra_runs $R/run_P8_s0,$R/run_P9_s0 > /home/intern_2603055/vvc/render_${c}_S015PP_ens.log 2>&1 || rm -rf $S/S015PP_ens/renders/$c
    elif [ -f $R/run_S015_s0/gaussians.pt ] && [ -f $R/run_P8_s0/gaussians.pt ] && [ ! -d $S/S015P_ens/renders/$c ] && [ ! -d $S/S015PP_ens/renders/$c ]; then
      while [ $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) -gt 36000 ]; do sleep 60; done
      echo "== render S015P $c $(date -u +%m-%d\ %H:%M)"
      /home/intern_2603055/vvc/ftg.sh $W/render_hidden.py $R/run_S015_s0 /home/intern_2603055/vvc/data/$c $S/S015P_ens/renders/$c --cull_near_frac 1.1 --cull_min_views 4 --extra_runs $R/run_P8_s0 > /home/intern_2603055/vvc/render_${c}_S015P_ens.log 2>&1 || rm -rf $S/S015P_ens/renders/$c
    fi
  done
  sleep 600
done
