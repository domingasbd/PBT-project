import numpy
from rdkit import Chem
from rdkit.Chem import rdchem
from rdkit.Chem import Fragments
from rdkit.Chem import rdMolDescriptors

def GetPrincipleQuantumNumber(atNum):
    """ Get principal quantum number for atom number """
    if atNum <= 2:
        return 1
    if atNum <= 10:
        return 2
    if atNum <= 18:
        return 3
    if atNum <= 36:
        return 4
    if atNum <= 54:
        return 5
    if atNum <= 86:
        return 6
    return 7

def EStateAll(mol, force=True):
    """ returns a tuple of EState indices for the molecule
        Reference: Hall, Mohney and Kier. JCICS *31* 76-81 (1991)
    """
    if not force and hasattr(mol, '_eStateIndices'):
        return mol._eStateIndices
    tbl = Chem.GetPeriodicTable()
    nAtoms = mol.GetNumAtoms()
    Is = numpy.zeros(nAtoms, dtype=numpy.float64)
    for i in range(nAtoms):
        at = mol.GetAtomWithIdx(i)
        d = at.GetDegree()
        if d > 0:
            atNum = at.GetAtomicNum()
            dv = tbl.GetNOuterElecs(atNum) - at.GetTotalNumHs()
            N = GetPrincipleQuantumNumber(atNum)
            Is[i] = (4. / (N * N) * dv + 1) / d
    dists = Chem.GetDistanceMatrix(mol, useBO=0, useAtomWts=0) + 1
    
    accum = numpy.zeros(nAtoms, dtype=numpy.float64)
    for i in range(nAtoms):
        for j in range(i + 1, nAtoms):
            p = dists[i, j]
            if p < 1e6:
                tmp = (Is[i] - Is[j]) / (p * p)
                accum[i] += tmp
                accum[j] -= tmp
    
    res = accum + Is
    mol._eStateIndices = res
    return res, Is, accum

def MaxISDiff(mol):
    EStates, IEStates, ISDiff = EStateAll(mol)
    return max(ISDiff)

def CountMultipleBonds(mol):
    nBM = 0
    for bond in mol.GetBonds():
        if rdchem.Bond.GetBondTypeAsDouble(bond) > 1:
            nBM = nBM + 1
    return nBM

def CalcPBT(mol): 
    """
    Calculate PBT (Persistence, Bioaccumulation, and Toxicity) related descriptors and final score
    
    Parameters:
    mol (rdkit.Chem.rdchem.Mol): RDKit molecule object
    
    Returns:
    tuple: (xN, HBD, nBM, MAXDP, PBT)
        xN: Number of halogens
        HBD: Number of Lipinski H-bond donors
        nBM: Number of multiple bonds
        MAXDP: Maximum E-State difference
        PBT: Final PBT score
    """
    xN = Fragments.fr_halogen(mol)
    
    HBD = rdMolDescriptors.CalcNumLipinskiHBD(mol)
    
    nBM = CountMultipleBonds(mol)
    
    MAXDP = MaxISDiff(mol)
    
    PBT = -1.5 + 0.64*xN + 0.22*nBM + -0.39*HBD + -0.062*MAXDP 
    return xN, HBD, nBM, MAXDP, PBT