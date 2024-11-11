
from rdkit import Chem
from rdkit.Chem import rdmolops

def neutralize_atoms(mol):
    if not mol:
        return None
    pattern = Chem.MolFromSmarts("[+1!h0!$([*]~[-1,-2,-3,-4]),-1!$([*]~[+1,+2,+3,+4])]")
    at_matches = mol.GetSubstructMatches(pattern)
    at_matches_list = [y[0] for y in at_matches]
    try:
        if len(at_matches_list) > 0:
            for at_idx in at_matches_list:
                atom = mol.GetAtomWithIdx(at_idx)
                chg = atom.GetFormalCharge()
                hcount = atom.GetTotalNumHs()
                atom.SetFormalCharge(0)
                atom.SetNumExplicitHs(hcount - chg)
                atom.UpdatePropertyCache()
    except:
        return mol
    return mol

def get_largest_frag(mol):
    if not mol:
        return None
    mol_frags = rdmolops.GetMolFrags(mol, asMols=True)
    largest_mol = max(mol_frags, default=mol, key=lambda m: m.GetNumAtoms())
    Chem.RemoveStereochemistry(largest_mol)
    return largest_mol
