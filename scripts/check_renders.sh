#!/bin/bash
# Report any render directory that is not a complete set of 8 views with the expected frame count.
cd /home/intern_2603055/vvc
for r in $(ls submissions | grep -v '\.zip\|\.manifest\|selfcap'); do
  [ -d submissions/$r/renders ] || continue
  for c in 004_1_seq0 006_1_seq0 007_0_seq0 009_0_seq0 011_0_seq0; do
    d=submissions/$r/renders/$c; [ -d $d ] || continue
    need=$(( ( $(ls data/$c/images/$(ls data/$c/images | head -1) | wc -l) + 9 ) / 10 ))
    nv=$(ls $d | wc -l); bad=0
    for v in $(ls $d); do [ $(ls $d/$v | wc -l) -ge $need ] || bad=1; done
    if [ "$nv" -lt 8 ] || [ $bad = 1 ]; then echo "BROKEN  $r/$c ($nv views, need ${need}/view)"; else echo "ok      $r/$c"; fi
  done
done
