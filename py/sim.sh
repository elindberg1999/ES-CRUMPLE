#!/bin/bash
#
# Run a single MD simulation and post-process dihedrals
# Assumes repository directory structure is intact
#

set -euo pipefail

if [ "$#" -ne 8 ]; then
    echo "Usage: $0 <psf> <dcd> <round> <idx> <conf> <init> <dih> <bools>"
    exit 1
fi

psf="$1"
dcd="$2"
round="$3"
idx="$4"
conf="$5"
init="$6"
dih="$7"
bools="$8"

echo "Input file is $conf"

# 1. Run MD simulation
echo "$(date) Running simulation (round=$round, idx=$idx)"

pmemd.hip -O \
    -i "$conf" \
    -o "../Simulations/$round/$idx/out.$round.$idx" \
    -p "$psf" \
    -c "../Simulations/$round/$idx/$init" \
    -x "../Simulations/$round/$idx/$dcd" \
    -inf "../Simulations/$round/$idx/info.$round.$idx"

echo "$(date) Simulation done. Post-processing..."

# 2. Post-process simulation
python postprocess_dihedrals_clean.py \
    "$psf" "$dcd" "$round" "$idx" "$dih" "$bools" \
    >> "logfiles/postprocess.log.$round.$idx"

echo "$(date) Finished round=$round idx=$idx"