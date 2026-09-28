#!/bin/bash
#SBATCH --job-name=c43_011b
#SBATCH --account=GOV114009
#SBATCH --partition=8gpus
#SBATCH --exclude=25a-hgpn144,25a-hgpn111
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --time=03:00:00
#SBATCH --output=/work/doreen071/vvc/jobs/c43_011b_%j.log
# Per-scene diagnosis on C35: 011 is the FG-SSIM drag (-0.018 vs the scene average). The five 011-only models
# from wave20 (pseudo weight 0.4-0.6, gating, full-res) were rendered and never used. Build a 011-only person
# source from them plus the ten new-recipe models (15 members), run the usual chain on that one scene, and
# assemble C43 = C35 with its 011 replaced. Per-scene scoring then isolates the effect.
V=/work/doreen071/vvc; W=$V/work; S=$V/submissions; PY=/home/doreen071/.conda/envs/vvc/bin/python
export OMP_NUM_THREADS=10 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cnt() { find $1 -name '*.jpg' 2>/dev/null | wc -l; }
W20="$S/M_P15_011w50/renders:1 $S/M_P16_011w60/renders:1 $S/M_P17_011gate/renders:1 $S/M_P18_011fr/renders:1 $S/M_P19_011damp/renders:1"
NEW="$S/M_S5PS/renders:1 $S/M_S5PS_s1/renders:1 $S/M_S5PS_w30/renders:1 $S/M_S5PS_w40/renders:1 $S/M_S5PS_gate/renders:1 $S/M_S5PS_s2/renders:1 $S/M_S5PS_fg80/renders:1 $S/M_S5PS_fg60/renders:1 $S/M_S5PS_s3/renders:1 $S/M_S5PS_w40s1/renders:1"
P=$S/S5P_011b/renders
$PY $W/weighted_ensemble.py $P $W20 $NEW --procs 90      # first source has only 011 -> restricts the walk to 011
echo "S5P_011b: $(cnt $P)/360"
NB=$S/NB3_011b/renders
$PY $W/weighted_ensemble.py $NB $P:4 $S/SIX_ens/renders:2 $S/M_DFX_s0/renders:2 $S/M_DFXFG_L60K/renders:1 $S/M_DFXFG_L60K_s1/renders:1 $S/M_S3_FGW40_L60K/renders:2 --procs 90
HB=$S/HB3_011b/renders
$PY $W/tta_average.py $HB $P $NB --procs 90
echo "HB3_011b: $(cnt $HB)/360"
for s in "" "320,288" "640,192"; do
  case "$s" in "") vn=DFX_011b;; "320,288") vn=DFX_011b_s1;; *) vn=DFX_011b_s2;; esac
  for i in 0 1 2 3 4 5 6 7; do CUDA_VISIBLE_DEVICES=$i DIFIX_SHARD=$i/8 $PY $W/difix_submission_h200.py $HB $S/$vn/renders ${s:+--shift $s} > $V/jobs/c43_${vn}_$i.log 2>&1 & done; wait
done
TTA=$S/TTA_011b/renders
$PY $W/tta_average.py $TTA $S/DFX_011b/renders $S/DFX_011b_s1/renders $S/DFX_011b_s2/renders --procs 90
echo "TTA_011b: $(cnt $TTA)/360"
RAW=$S/C43_011b_raw/renders
CUDA_VISIBLE_DEVICES=0 $PY $W/composite2.py tree $HB $TTA 0.85 $S/BGT_new/renders 1.0 $RAW --cases 011_0_seq0 --feather 101 > $V/jobs/cand_C43_011.log 2>&1
echo "C43 011 raw: $(cnt $RAW)/360"
# colour fix on 011 (taper 1.2) straight from the tree: pack the 011 tree to a temp zip so camcolor2 can read it
$PY - <<'PYZ'
import zipfile, os
root='/work/doreen071/vvc/submissions/C43_011b_raw/renders'
with zipfile.ZipFile('/work/doreen071/vvc/submissions/C43_tmp011.zip','w',zipfile.ZIP_STORED) as z:
    z.writestr('team_name.txt','Doreen071')
    for v in sorted(os.listdir(f'{root}/011_0_seq0')):
        for f in sorted(os.listdir(f'{root}/011_0_seq0/{v}')):
            z.write(f'{root}/011_0_seq0/{v}/{f}', f'sparseViewTrack/renders/011_0_seq0/{v}/{f}')
print('tmp zip ok')
PYZ
OUT=$S/C43_011b/renders; rm -rf $OUT; mkdir -p $OUT
for i in 0 1 2 3 4 5 6 7; do $PY $W/camcolor2.py apply $S/C43_tmp011.zip $OUT 011_0_seq0 $W/priors/cc2_012_0.json taper 1.2 --shard $i/8 --tau 0.10 & done; wait
echo "C43 011 fixed: $(cnt $OUT)/360"
# assemble: C35's other four scenes + boosted 011
rm -rf $S/C43_asm; mkdir -p $S/C43_asm/_x
$PY -c "import zipfile; zipfile.ZipFile('$S/C35_p22_both.zip').extractall('$S/C43_asm/_x')"
mv $S/C43_asm/_x/sparseViewTrack/renders $S/C43_asm/renders; rm -rf $S/C43_asm/_x
rm -rf $S/C43_asm/renders/011_0_seq0; cp -al $OUT/011_0_seq0 $S/C43_asm/renders/011_0_seq0
n=$(cnt $S/C43_asm/renders); echo "-- C43_011b assembled: $n/2056"
[ "$n" -eq 2056 ] && $PY $W/pack_submission.py $S/C43_011b.zip $S/C43_asm/renders && chmod 444 $S/C43_011b.zip && echo "PACKED C43_011b $(date +%H:%M)"
rm -rf $S/C43_asm $S/C43_tmp011.zip
echo C43_DONE $(date +%H:%M)
