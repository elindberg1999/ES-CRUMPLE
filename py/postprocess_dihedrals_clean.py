import sys
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import MDAnalysis as mda
from MDAnalysis.analysis import pca, align, dihedrals
from MDAnalysis.analysis.base import AnalysisBase, AnalysisFromFunction, analysis_class

warnings.filterwarnings('ignore')

# === Command-line argument parsing ===
if len(sys.argv) < 7:
    print("Usage: postprocess.py <parameter file> <trajectory> <round index> <simulation index> <indices file> <bools file>")
    sys.exit(1)

parm_file = sys.argv[1]
traj_file = sys.argv[2]
round_idx = int(sys.argv[3])
sim_idx = int(sys.argv[4])
indices_path = sys.argv[5]
bools_path = sys.argv[6]

# === Helper Functions ===
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def squish_cos(angle, b):
    norm_factor = sigmoid(b) - sigmoid(-b)
    return (sigmoid(-b * np.cos(np.pi * angle / 180)) - sigmoid(-b)) / norm_factor

def squish_sin(angle, b):
    norm_factor = sigmoid(b) - sigmoid(-b)
    return (sigmoid(-b * np.sin(np.pi * angle / 180)) - sigmoid(-b)) / norm_factor

# === Load inputs ===
b = 5
dih_idxs = np.load(indices_path, allow_pickle=True)
meth_bools = np.load(bools_path, allow_pickle=True)

print(dih_idxs, meth_bools)
print("meth_bools length:", len(meth_bools))

# === Determine input size ===
if meth_bools[0] == -1:
    dihs = dih_idxs
    n_dih = len(dihs)
    input_size = n_dih * 2
else:
    dihs = dih_idxs
    n_dih = len(dih_idxs.reshape(-1, dih_idxs.shape[-1]))
    input_size = n_dih * 2 - len(meth_bools) - len([a for a in meth_bools if a != True])

print("Number of dihedrals:", n_dih)
print("Input size:", input_size)

# === Load simulation data ===
traj_path = f'../Simulations/{round_idx}/{sim_idx}/{traj_file}'
u = mda.Universe(parm_file, traj_path, format="NCDF")
#u = mda.Universe(parm_file)

# === Prepare output array ===
dih_a = np.zeros((len(u.trajectory), input_size + 3))


# === Process frames ===
for i, ts in enumerate(u.trajectory):
    dih_list = [round_idx, sim_idx, i]

    for idx, d in enumerate(dihs):
        if meth_bools[0] == -1:
            sel_str = 'index ' + ' '.join(map(str, d))
            atoms = u.select_atoms(sel_str)
            dih_val = atoms.dihedral.value()
            dih_list += [squish_cos(dih_val, b), squish_sin(dih_val, b)]
        else:
            phi_sel = u.select_atoms('index ' + ' '.join(map(str, d[0])))
            psi_sel = u.select_atoms('index ' + ' '.join(map(str, d[1])))
            omega_sel = u.select_atoms('index ' + ' '.join(map(str, d[2])))

            dih_list += [
                squish_cos(phi_sel.dihedral.value(), b),
                squish_sin(phi_sel.dihedral.value(), b),
                squish_cos(psi_sel.dihedral.value(), b),
                squish_sin(psi_sel.dihedral.value(), b)
            ]

            if meth_bools[idx]:
                dih_list.append(squish_cos(omega_sel.dihedral.value(), b))

    dih_a[i] = np.array(dih_list)

# === Save outputs ===
np.save(f'../Simulations/data/{round_idx}_{sim_idx}.npy', dih_a)

if not os.path.exists('../Simulations/data/input_size.npy'):
    np.save('../Simulations/data/input_size.npy', input_size)
else:
    print(f"The file '../Simulations/data/input_size.npy' exists.")

#np.save('test.npy', dih_a)
#if not os.path.exists('test2.npy'):
#    np.save('test2.npy', input_size)
#else:
#    print(f"The file 'test2' exists.")