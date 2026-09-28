#!/bin/bash
# wait for the SSIM-member probe (training + 3 evals) to finish, then run the Difix fine-tune (needs ~41 GB GPU)
while ! grep -q "SSIM MEMBER PROBE DONE\|FAILED" /home/intern_2603055/vvc/ssim_member.log; do sleep 120; done
echo "probe finished $(date -u +%H:%M); starting Difix fine-tune"
/home/intern_2603055/vvc/work/finetune_difix.sh
