#!/bin/bash

#
# Convert NetCDF trajectories to Amber ASCII restart files (.rst)
# Usage:
#   ./convToAmber.sh <psf> <round> <n_sims> <coord_filename>
#

psf=$1
round=$2
sims=$3
coord=$4

cd "$(dirname "$0")"

for ((i=0;i<=sims;i++)); do

cd ../Simulations/$((round + 1))/$i
cpptraj -p ../Simulations/$psf <<EOF
trajin ncdf_$coord
trajout $coord restart
go
EOF
done

