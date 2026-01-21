import sys
import os

import numpy as np
import networkx as nx
import MDAnalysis as mda
from MDAnalysis.topology.guessers import guess_bonds
from MDAnalysis.lib.distances import distance_array
from MDAnalysis.core.universe import Merge

from rdkit import Chem
from rdkit.Chem import AllChem

from graphein.molecule import construct_graph, MoleculeGraphConfig
from graphein.molecule.edges.atomic import add_atom_bonds

if os.path.exists('bools.npy'):
    os.remove('bools.npy')

if os.path.exists('dih.npy'):
    os.remove('dih.npy')

if len(sys.argv) < 2:
    print("Usage: python script.py <pdb_file>")
    sys.exit(1)

# --- Load molecule from PDB using RDKit ---
pdb_fname = sys.argv[1]
rdmol = Chem.MolFromPDBFile(pdb_fname, removeHs=False)

if rdmol is None:
    raise ValueError("RDKit failed to load the PDB file.")

# Add hydrogens and generate 3D conformer
rdmol = Chem.AddHs(rdmol)
params = AllChem.ETKDGv3()
params.randomSeed = 0xf00d

if AllChem.EmbedMolecule(rdmol, params) != 0:
    raise RuntimeError("Conformer generation failed after adding hydrogens.")

# Optionally optimize geometry
AllChem.UFFOptimizeMolecule(rdmol)

# Sanity check
conf = rdmol.GetConformer()
assert conf.GetNumAtoms() == rdmol.GetNumAtoms(), "Mismatch in atom count and coordinates!"

# --- Construct molecular graph and find largest ring ---
config = MoleculeGraphConfig(
    add_hs=True,
    generate_conformer=False,
    edge_construction_functions=[add_atom_bonds],
    node_metadata_functions=[]
)
graph = construct_graph(mol=rdmol, config=config)
cycles = nx.cycle_basis(graph)

if not cycles:
    raise ValueError("No rings found in the molecule.")

largest_cycle = max(cycles, key=len)
ring_atom_indices = [int(node.split(":")[1]) for node in largest_cycle]
print("Atom indices in largest ring:", ring_atom_indices)

# --- Load structure using MDAnalysis ---
u = mda.Universe(pdb_fname)
positions = u.atoms.positions
types = u.atoms.types

# Guess bonds
distance_array(positions, positions, box=u.dimensions)
bond_array = guess_bonds(u.atoms, coords=positions)
bond_list = [(int(i), int(j)) for i, j in bond_array]

# Reconstruct universe with guessed bonds
u = Merge(u.atoms)
u.atoms.positions = positions
u.add_TopologyAttr('bonds', bond_list)
u_all = u.select_atoms('all')

# Select atoms in ring
ring_selection = "index " + " ".join(map(str, ring_atom_indices))
ring_atoms = u.select_atoms(ring_selection)

# --- Determine loop type ---
mode = 0 if set(ring_atoms.names) == {'C', 'CA', 'N'} else 1

if mode == 0:
    print("Loop contains only backbone atoms.")
else:
    print("Largest loop does not contain only backbone atoms. Entering mode 2.")
    print(ring_atoms.names)

# --- Mode 0: Backbone dihedrals calculation ---
if mode == 0:
    # Rotate list so smallest index is first
    min_index = ring_atom_indices.index(min(ring_atom_indices))
    shifted = ring_atom_indices[min_index:] + ring_atom_indices[:min_index]
    grouped_indices = [sorted(shifted[i:i+3]) for i in range(0, len(shifted), 3)]
    grouped_atoms = [ring_atoms[i:i+3] for i in range(0, len(ring_atoms), 3)]

    # Assign residue IDs
    for j, group in enumerate(grouped_indices):
        for res in grouped_atoms:
            if set(res.indices) == set(group):
                res.residues.resids = j + 1

    max_res_id = max(ring_atoms.resids)
    phis, psis, omegas, methylated_flags = [], [], [], []

    # Compute torsion angle atom indices
    for k in range(1, max_res_id + 1):
        # Get current, previous, and next residue groups
        this_res = next(g for g in grouped_atoms if k in g.resids)
        prev_res = next(g for g in grouped_atoms if (k - 1 if k > 1 else max_res_id) in g.resids)
        next_res = next(g for g in grouped_atoms if (k + 1 if k < max_res_id else 1) in g.resids)

        # Atom indices
        prev_CA = prev_res.select_atoms('name CA')[0].index
        prev_C  = prev_res.select_atoms('name C')[0].index
        N       = this_res.select_atoms('name N')[0].index
        CA      = this_res.select_atoms('name CA')[0].index
        C       = this_res.select_atoms('name C')[0].index
        next_N  = next_res.select_atoms('name N')[0].index

        # Methylation check
        try:
            bonded_atoms = [a.index for b in u_all[N].bonds for a in b.atoms if a != u_all[N]]
        except mda.exceptions.NoDataError:
            bonded_atoms = []

        methylated = 'H' not in u_all[bonded_atoms].elements
        methylated_flags.append(int(methylated))

        # Collect torsion atoms
        omegas.append([prev_CA, prev_C, N, CA])
        phis.append([prev_C, N, CA, C])
        psis.append([N, CA, C, next_N])

    # Filter and print non-methylated residues
    dihedrals = np.array([[phi, psi, omega] for phi, psi, omega in zip(phis, psis, omegas)])
    methylated_flags = np.array(methylated_flags, dtype=bool)

    np.save("../Simulations/data/dih.npy", dihedrals)
    np.save("../Simulations/data/bools.npy", methylated_flags)

# --- Mode 1: General ring torsion calculation ---
else:
    dihedrals = []

    for i in range(len(ring_atom_indices) - 3):
        dihedrals.append(ring_atom_indices[i:i+4])

    # Wrap-around dihedrals for ring closure
    dihedrals.append([ring_atom_indices[-3], ring_atom_indices[-2], ring_atom_indices[-1], ring_atom_indices[0]])
    dihedrals.append([ring_atom_indices[-2], ring_atom_indices[-1], ring_atom_indices[0], ring_atom_indices[1]])
    dihedrals.append([ring_atom_indices[-1], ring_atom_indices[0], ring_atom_indices[1], ring_atom_indices[2]])

    dih_np = np.array(dihedrals)
    np.save("../Simulations/data/dih.npy", dih_np)
    np.save("../Simulations/data/bools.npy", np.array([-1]))
