#!/bin/bash
#SBATCH --job-name=c30_fgheavy
#SBATCH --account=GOV114009
#SBATCH --partition=8gpus
#SBATCH --exclude=25a-hgpn144,25a-hgpn111
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --time=06:00:00
#SBATCH --output=/work/doreen071/vvc/jobs/c30_fgheavy_%j.log
# FG-SSIM is the one metric that decides 2nd place (+0.0139 needed) and the record says FGW80 had the highest
# FG-SSIM of any solo model. wave25 is training fg60/fg80 with stride-5 points and pseudo views, i.e. the best
# FG recipe we know crossed with the two levers that already worked. Rather than diluting them into a general
# mix, this builds a person source out of the FG-heavy models ONLY (fg80 + fg60, plus the fg-4.0 S5PS trio as
# the stabiliser), which is the most FG-SSIM-targeted source we can construct.
module load cuda/12.6 2>/dev/null
V=/work/doreen071/vvc; W=$V/work; S=$V/submissions; PY=/home/doreen071/.conda/envs/vvc/bin/python
export OMP_NUM_THREADS=10 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cnt() { find $1 -name '*.jpg' 2>/dev/null | wc -l; }
CASES="004_1_seq0 006_1_seq0 007_0_seq0 009_0_seq0 011_0_seq0"
NEWM="S5PS_fg80 S5PS_fg60 S5PS_s3 S5PS_w40s1"
echo "== waiting for wave24/25 training, $(date +%H:%M) =="
for m in $NEWM; do for c in $CASES; do until [ -f $V/runs/siga_$c/run_$m/gaussians.pt ]; do sleep 60; done; done; echo "  $m trained $(date +%H:%M)"; done
render() {
  OUT=$S/M_$1/renders/$2
  [ "$(cnt $OUT)" -gt 0 ] && return
  TD=$(mktemp -d /tmp/tex.XXXXXX)/py312_cu121; mkdir -p $TD; cp -r $V/envcache/gsplat_prebuilt/gsplat_cuda $TD/
  PATH=/home/doreen071/.conda/envs/vvc312/bin:$PATH TORCH_CUDA_ARCH_LIST=9.0 TORCH_EXTENSIONS_DIR=$TD CUDA_VISIBLE_DEVICES=$3 python $W/render_hidden.py $V/runs/siga_$2/run_$1 $V/data/$2 $OUT --cull_near_frac 1.1 --cull_min_views 4 > $V/jobs/renderM_$1_$2.log 2>&1
  rm -rf $(dirname $TD)
}
echo "== render, $(date +%H:%M) =="
k=0; JOBS=()
for m in $NEWM; do for c in $CASES; do JOBS+=("$m $c"); done; done
for j in "${JOBS[@]}"; do render $j $((k % 8)) & k=$((k+1)); [ $((k % 8)) -eq 0 ] && wait; done; wait
for m in $NEWM; do echo "M_$m: $(cnt $S/M_$m)/2056"; done
# FG-heavy person branch: the two high-FG-weight configs carry the most weight, the fg=4 trio stabilises
P=$S/S5P_FGH/renders
$PY $W/weighted_ensemble.py $P $S/M_S5PS_fg80/renders:3 $S/M_S5PS_fg60/renders:3 $S/M_S5PS/renders:1 $S/M_S5PS_s1/renders:1 $S/M_S5PS_w30/renders:1 $S/M_S5PS_w40/renders:1 $S/M_S5PS_gate/renders:1 $S/M_S5PS_s2/renders:1 $S/M_S5PS_s3/renders:1 $S/M_S5PS_w40s1/renders:1 --procs 90
echo "S5P_FGH: $(cnt $P)/2056"
NB=$S/NB3_FGH/renders
$PY $W/weighted_ensemble.py $NB $S/SIX_ens/renders:2 $S/M_DFX_s0/renders:2 $S/M_DFXFG_L60K/renders:1 $S/M_DFXFG_L60K_s1/renders:1 $P:4 $S/M_S3_FGW40_L60K/renders:2 --procs 90
HB=$S/HB3_FGH/renders
$PY $W/tta_average.py $HB $P $NB --procs 90
echo "HB3_FGH: $(cnt $HB)/2056"
for s in "" "320,288" "640,192"; do
  case "$s" in "") vn=DFX_FGH;; "320,288") vn=DFX_FGH_s1;; *) vn=DFX_FGH_s2;; esac
  echo "== difix $vn, $(date +%H:%M) =="
  for i in 0 1 2 3 4 5 6 7; do CUDA_VISIBLE_DEVICES=$i DIFIX_SHARD=$i/8 $PY $W/difix_submission_h200.py $HB $S/$vn/renders ${s:+--shift $s} > $V/jobs/fgh_${vn}_$i.log 2>&1 & done; wait
done
TTA=$S/TTA_FGH/renders
$PY $W/tta_average.py $TTA $S/DFX_FGH/renders $S/DFX_FGH_s1/renders $S/DFX_FGH_s2/renders --procs 90
NAME=C30_fgheavy; RAW=$S/${NAME}_raw/renders; i=0
for c in $CASES; do
  CUDA_VISIBLE_DEVICES=$i $PY $W/composite2.py tree $HB $TTA 0.85 $S/BGX_tp15/renders 1.0 $RAW --cases $c --feather 101 --q 92 > $V/jobs/cand_${NAME}_$c.log 2>&1 &
  i=$((i+1))
done; wait
n=$(cnt $RAW); echo "-- ${NAME}_raw: $n/2056"; [ "$n" -eq 2056 ] || exit 1
$PY $W/pack_submission.py $S/${NAME}_raw.zip $RAW && chmod 444 $S/${NAME}_raw.zip
OUT=$S/${NAME}/renders; rm -rf $S/${NAME}/_x $OUT; mkdir -p $S/${NAME}/_x $OUT
$PY -c "import zipfile; zipfile.ZipFile('$S/${NAME}_raw.zip').extractall('$S/${NAME}/_x')"
mv $S/${NAME}/_x/sparseViewTrack/renders/* $OUT/; rm -rf $S/${NAME}/_x
rm -rf $OUT/011_0_seq0 $OUT/007_0_seq0
for i in 0 1 2 3 4 5 6 7; do $PY $W/camcolor2.py apply $S/${NAME}_raw.zip $OUT 011_0_seq0 $W/priors/cc2_012_0.json taper 1.2 --shard $i/8 --tau 0.10 & done; wait
for i in 0 1 2 3 4 5 6 7; do $PY $W/camcolor2.py apply $S/${NAME}_raw.zip $OUT 007_0_seq0 $W/priors/cc2_001_1.json gain 1.0 --shard $i/8 & done; wait
n2=$(cnt $OUT); echo "-- ${NAME}: $n2/2056"
[ "$n2" -eq 2056 ] && $PY $W/pack_submission.py $S/${NAME}.zip $OUT && chmod 444 $S/${NAME}.zip && echo "PACKED ${NAME} $(date +%H:%M)"
echo C30_DONE $(date +%H:%M)
