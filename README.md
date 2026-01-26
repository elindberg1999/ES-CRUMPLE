# ES-CRUMPLE – Enhanced Sampling by Clustering and Rare-state Upsampling of Macrocyclic Peptides from a Learned Encoding

> **Disclaimer:** This application uses modified code from [EasyDeepDriveMD](https://github.com/darrenjhsu/EasyDeepDriveMD) by Dr. Darren Hsu.
>
>Modifications include:
>
>- main.sh: Added calls to new features and added support for AMBER
>- train.py: Changed the pipeline to take postprocessed dihedral angles as input
>- modelAE.py: Changed autoencoder architecture to accommodate dihedral angle data
>- sim.sh: Changed the MD call to pmemd.hip (will need to be changed to your simulation engine)

---

<p align="center">
  <img src="https://i.ibb.co/CpJkDMqV/ES-CRUMPLE-ghv1-01.png" alt="Workflow" />
</p>

---

## Overview

ES-CRUMPLE is a Python-based workflow for **adaptive molecular dynamics (MD) sampling** using an **Autoencoder (AE)**.  

The workflow iteratively:

1. Runs MD simulations of a user-defined number of replicas.
2. Extracts **dihedral features** from the trajectories.
3. Trains an **AE** to embed conformations into a latent space.
4. Performs **clustering (DBSCAN)** in the latent space to identify **outlier conformations**.
5. Suggests new initial coordinates for the next round of simulations.

## Repository Structure

- ES-CRUMPLE/
  - py/
    - convToAmber.sh
    - main.sh
    - modelAE.py
    - parse_dihedrals_clean.py
    - postprocess_dihedrals_clean.py
    - sim.sh
    - train.py
    - latent_embeddings
  - Simulations/
    - data
    - saved_models
  - Structures/
  - Template/
  - README.md

> **Note:** Large trajectory files (`.ncdf`, `.mdcrd`) are **not included**. Example data is provided for testing the workflow.

## Installation

1. **Clone the repository**

```bash
git clone https://github.com/elindberg1999/ES-CRUMPLE.git
cd ES-CRUMPLE
```
2. **Create a python environment**
```bash
conda create -n escrumple python=3.10
conda activate escrumple
```
3. **Install dependencies**

```bash
pip install numpy pandas matplotlib tensorflow scikit-learn MDAnalysis
```
## Setup

> **Note:** You will need to change the file names in main.sh. "psf" should be your relative path to your parameter file, "template_folder" should keep all files necessary for the simulation to run (.mdin files and analogs in other software), "init_coord" should be just the file path to your strating structure, sim_config is the name of your .mdin file or equivalent.

Example:

```bash
# Source config
n_rounds=250
n_sims=48
psf=../Structures/CycAsol.parm7
template_folder=../Template/
init_coord=eq3.rst
sim_config=DDMD.in
# This has to match the output of your simulations
sample_dcd=DDMD.mdcrd
```


## Scripts Description

|   Script | Purpose  |
|----------|----------|
| main.sh | Main loop for iterative simulation and clustering |
| pdbizer.py | Creates solventless PDB to use for graph network in parse_dihedrals_clean.py |
| parse_dihedrals_clean.py | Parses atom indicies of largest cycle in the molecular graph (this is taken as the backbone) |
| sim.sh | Runs a single simulation, then calls postprocessing |
| postprocess_dihedrals_clean.py | Extracts dihedral angles from the simulation and outputs a numpy array |
| modelAE.py | Autoencoder architechure with variable input size |
| train.py | Trains the autoencoder and clusters the resulting embedding |

## References

- [EasyDeepDriveMD Repository](https://github.com/darrenjhsu/EasyDeepDriveMD) – Original adaptive sampling and CVAE workflow by Darren Hsu
- [MDAnalysis](https://www.mdanalysis.org) – Python library for molecular dynamics analysis
- [TensorFlow](https://www.tensorflow.org) – Machine learning framework used for the autoencoder
- [Scikit-learn DBSCAN](https://scikit-learn.org/stable/modules/clustering.html#dbscan) – Clustering algorithm for latent embeddings

- [Amber](https://ambermd.org) – Molecular dynamics engine for simulations




