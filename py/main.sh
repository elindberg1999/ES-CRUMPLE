#!/bin/bash

# The main workflow for adaptive sampling through AE

# Source config
n_rounds=250
n_sims=48
psf=PATH_TO_YOUR_PSF
template_folder=PATH_TO_YOUR_TEMPLATE
init_coord=INIT_COORD_FILE
sim_config=INIT_MD_FILE
# This has to match the output of your simulations
sample_dcd=SAMPLE_DCD_FILE

mkdir -p ../Simulations
mkdir -p ../Simulations/data

# TODO: determine the progress

restart_round=0
restart_stage=0
if test -f "../Simulations/progress"; then 
  source ../Simulations/progress
fi
echo round $restart_round stage $restart_stage

if test -f "../Simulations/data/dih.npy"; then 
  echo "Dihedral file exists."
else
  echo "Dihedral file does not exist. Creating bool file."
  srun --ntasks=1 --gpus=1 --gpu-bind=closest --exclusive --exact python pdbizer.py $psf ${template_folder}/${init_coord} > logfiles/pdbizer.log
  srun --ntasks=1 --gpus=1 --gpu-bind=closest --exclusive --exact python parse_dihedrals_clean.py ../Structures/nosolvent.pdb >& logfiles/parse.log
fi


for round in `seq $restart_round $n_rounds`
do
  echo -e "\n`date` We are in round ${round}!\n"

  if [ $round -ge $restart_round ] && [ $restart_stage -le 0 ] ; then
    # Stage 0
    mkdir -p ../Simulations/$round
    # Copy simulation config
    echo "`date` Copy sim config for round $round"
    for idx in `seq 0 $((n_sims -1))`
    do
      mkdir -p ../Simulations/$round/$idx
      cp ${template_folder}/${sim_config} ../Simulations/$round/$idx/
    done
     
    # Copy initial coordinates for round 0
    if [ $round -eq 0 ]; then
      echo "`date` Copy initial coord for round 0"
      for idx in `seq 0 $((n_sims -1))`
      do
        cp ${template_folder}/${init_coord} ../Simulations/$round/$idx/
      done
    fi

    restart_stage=1
    echo -e restart_round=$round > ../Simulations/progress 
    echo -e restart_stage=1 >> ../Simulations/progress
  else
    echo Skipping initial prep for round $round 
  fi

  if [ $round -ge $restart_round ] && [ $restart_stage -le 1 ]; then
    # Stage 1: Run simulation and post processing (sim.sh)
    echo "`date` Run simulations and post processing"
    for idx in `seq 0 $((n_sims - 1))`
    do
    srun --ntasks=1 --gpus=1 --gpu-bind=closest --exclusive --exact sim.sh $psf $sample_dcd $round $idx $sim_config $init_coord ../Simulations/data/dih.npy ../Simulations/data/bools.npy  &
    done
    wait
    
    restart_stage=2
    echo -e restart_round=$round > ../Simulations/progress 
    echo -e restart_stage=2 >> ../Simulations/progress
  else
    echo Skipping simulation and post-processing, for round $round
  fi

  if [ $round -ge $restart_round ] && [ $restart_stage -le 2 ]; then
    # Run training (AE), also suggests new initial coordinates
    echo "`date` Train AE and suggest new inits"
    srun --ntasks=1 --gpus=1 --gpu-bind=closest --exclusive --exact python train.py $round $n_sims $psf $sample_dcd $init_coord >& logfiles/train.$round.log
    sh convToAmber.sh $psf $round $((n_sims - 1)) $init_coord
  
    restart_stage=0
    echo -e restart_round=$((round+1)) > ../Simulations/progress
    echo -e restart_stage=0 >> ../Simulations/progress
  fi

  # Tidy up simulation data 

done
wait
