#!/bin/bash
#SBATCH --job-name=c39_trim
#SBATCH --account=GOV114009
#SBATCH --partition=8gpus
#SBATCH --exclude=25a-hgpn144,25a-hgpn111
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --time=05:00:00
#SBATCH --output=/work/doreen071/vvc/jobs/c39_trim_%j.log
# Same 10 new-recipe person models that feed C22/C30, but aggregated with a per-pixel TRIMMED mean (drop the
# 2 lowest and 2 highest of 10) instead of a plain mean: keeps the noise averaging, discards the member that
# put a floater or a limb in the wrong place at that pixel, so structure (SSIM) should blur less.
V=/work/doreen071/vvc; W=$V/work; S=$V/submissions; PY=/home/doreen071/.conda/envs/vvc/bin/python
export OMP_NUM_THREADS=10 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cnt() { find $1 -name '*.jpg' 2>/dev/null | wc -l; }
CASES="004_1_seq0 006_1_seq0 007_0_seq0 009_0_seq0 011_0_seq0"
NEW="M_S5PS M_S5PS_s1 M_S5PS_w30 M_S5PS_w40 M_S5PS_gate M_S5PS_s2 M_S5PS_fg80 M_S5PS_fg60 M_S5PS_s3 M_S5PS_w40s1"
for m in $NEW; do until [ "$(cnt $S/$m)" -eq 2056 ]; do sleep 60; done; done
echo "== all 10 member renders present, $(date +%H:%M) =="
P=$S/S5P_trim/renders
$PY $W/weighted_ensemble_trim.py $P $(for m in $NEW; do printf "%s " $S/$m/renders; done) --trim 2 --procs 90
echo "S5P_trim: $(cnt $P)/2056"
NB=$S/NB3_trim/renders
$PY $W/weighted_ensemble.py $NB $S/SIX_ens/renders:2 $S/M_DFX_s0/renders:2 $S/M_DFXFG_L60K/renders:1 $S/M_DFXFG_L60K_s1/renders:1 $P:4 $S/M_S3_FGW40_L60K/renders:2 --procs 90
HB=$S/HB3_trim/renders
$PY $W/tta_average.py $HB $P $NB --procs 90
for s in "" "320,288" "640,192"; do
  case "$s" in "") vn=DFX_trim;; "320,288") vn=DFX_trim_s1;; *) vn=DFX_trim_s2;; esac
  echo "== difix $vn, $(date +%H:%M) =="
  for i in 0 1 2 3 4 5 6 7; do CUDA_VISIBLE_DEVICES=$i DIFIX_SHARD=$i/8 $PY $W/difix_submission_h200.py $HB $S/$vn/renders ${s:+--shift $s} > $V/jobs/trim_${vn}_$i.log 2>&1 & done; wait
done
TTA=$S/TTA_trim/renders
$PY $W/tta_average.py $TTA $S/DFX_trim/renders $S/DFX_trim_s1/renders $S/DFX_trim_s2/renders --procs 90
for V2 in "C39_trim:BGX_tp15" "C39_trim_bgnew:BGX_new"; do
  NAME=${V2%%:*}; BG=$S/${V2#*:}/renders
  RAW=$S/${NAME}_raw/renders; i=0
  for c in $CASES; do
    CUDA_VISIBLE_DEVICES=$i $PY $W/composite2.py tree $HB $TTA 0.85 $BG 1.0 $RAW --cases $c --feather 101 > $V/jobs/cand_${NAME}_$c.log 2>&1 &
    i=$((i+1))
  done; wait
  n=$(cnt $RAW); echo "-- ${NAME}_raw: $n/2056"; [ "$n" -eq 2056 ] || continue
  $PY $W/pack_submission.py $S/${NAME}_raw.zip $RAW && chmod 444 $S/${NAME}_raw.zip
  OUT=$S/${NAME}/renders; rm -rf $S/${NAME}/_x $OUT; mkdir -p $S/${NAME}/_x $OUT
  $PY -c "import zipfile; zipfile.ZipFile('$S/${NAME}_raw.zip').extractall('$S/${NAME}/_x')"
  mv $S/${NAME}/_x/sparseViewTrack/renders/* $OUT/; rm -rf $S/${NAME}/_x
  rm -rf $OUT/011_0_seq0 $OUT/007_0_seq0
  for i in 0 1 2 3 4 5 6 7; do $PY $W/camcolor2.py apply $S/${NAME}_raw.zip $OUT 011_0_seq0 $W/priors/cc2_012_0.json taper 1.2 --shard $i/8 --tau 0.10 & done; wait
  for i in 0 1 2 3 4 5 6 7; do $PY $W/camcolor2.py apply $S/${NAME}_raw.zip $OUT 007_0_seq0 $W/priors/cc2_001_1.json gain 1.0 --shard $i/8 & done; wait
  n2=$(cnt $OUT); echo "-- ${NAME}: $n2/2056"
  [ "$n2" -eq 2056 ] && $PY $W/pack_submission.py $S/${NAME}.zip $OUT && chmod 444 $S/${NAME}.zip && echo "PACKED ${NAME} $(date +%H:%M)"
done
echo C39_DONE $(date +%H:%M)
