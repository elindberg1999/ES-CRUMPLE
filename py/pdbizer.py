import parmed as pmd
import sys
import os

if len(sys.argv) < 3:
    print("Usage: python pdbizer.py <topology_file.parm7> <coordinates_file.rst7>")
    sys.exit(1)

top_file = sys.argv[1]
coord_file = sys.argv[2]

try:
    structure = pmd.load_file(top_file, coord_file)
except Exception as e:
    print(f"Error loading files: {e}")
    sys.exit(1)

residues_to_remove = {'HOH', 'WAT', 'DMSO', 'MOH'}

filtered_structure = pmd.Structure()
filtered_structure.box = structure.box

atom_indices = []

for i, atom in enumerate(structure.atoms):
    if atom.residue.name not in residues_to_remove:
        filtered_structure.add_atom(atom, atom.residue.name, atom.residue.number)
        atom_indices.append(i)

if structure.coordinates is not None:
    coords = structure.coordinates
    if coords.ndim == 3:
        filtered_structure.coordinates = coords[:, atom_indices, :]
    elif coords.ndim == 2:
        filtered_structure.coordinates = coords[atom_indices, :]
    else:
        raise ValueError(f"Unexpected coordinates shape: {coords.shape}")

output_dir = os.path.dirname(coord_file)
input_basename = os.path.splitext(os.path.basename(coord_file))[0]
output_pdb = os.path.join(output_dir, input_basename + "_nosolvent.pdb")

print(f"Saving cleaned PDB to: ../Structures/nosolvent.pdb")
filtered_structure.save('../Structures/nosolvent.pdb')