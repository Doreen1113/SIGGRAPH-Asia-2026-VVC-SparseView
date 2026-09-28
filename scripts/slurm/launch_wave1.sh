#!/bin/bash
# Wave 1. val 001_1: reproduce reference members (FULLRES_s0,J_s0,GATE_s0,G4M_s0) + candidates; test scenes: candidates only.
set -u
SB=/work/doreen071/vvc/work/slurm/train_member.sbatch
sub() { local M=$1 C=$2; shift 2
  [ -f /work/doreen071/vvc/runs/siga_$C/run_$M/gaussians.pt ] && { echo "skip $M/$C"; return; }
  sbatch --job-name=${M}_${C} --export=ALL,MEMBER=$M,CASE=$C,POINTS_OWNER=0,OVR="$*" $SB; }
V=001_1_seq0
sub FULLRES_s0 $V data.scale=1.0
sub J_s0       $V data.scale=0.5
sub GATE_s0    $V model.marginal_gating=true
sub G4M_s0     $V init.num_gaussians=4000000
sub G3M_s0     $V init.num_gaussians=3000000
for C in $V 004_1_seq0 006_1_seq0 007_0_seq0 009_0_seq0 011_0_seq0; do
  sub FULLRES_s1 $C data.scale=1.0 train.seed=1
  sub GATE_FR_s0 $C data.scale=1.0 model.marginal_gating=true
  sub G4M_FR_s0  $C data.scale=1.0 init.num_gaussians=4000000
done
squeue -u doreen071 -h -o "%i %j %T" | sort -k2 | awk '{print $2, $3}' | tr '\n' ';'
