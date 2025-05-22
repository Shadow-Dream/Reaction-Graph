from chemprop.featurizers.atom import MultiHotAtomFeaturizer
from chemprop.featurizers.bond import MultiHotBondFeaturizer
from chemprop.featurizers.molgraph.reaction import CondensedGraphOfReactionFeaturizer

def get_featurizor(metadata):
    atomic_nums = metadata["atomic_nums"]
    degrees = metadata["degrees"]
    formal_charges = metadata["formal_charges"]
    chiral_tags = metadata["chiral_tags"]
    num_Hs = metadata["num_Hs"]
    hybridizations = metadata["hybridizations"]
    bond_types = metadata["bond_types"]
    stereos = metadata["stereos"]
    atom_featurizer = MultiHotAtomFeaturizer(atomic_nums, degrees, formal_charges, chiral_tags, num_Hs, hybridizations)
    bond_featurizer = MultiHotBondFeaturizer(bond_types, stereos)
    return CondensedGraphOfReactionFeaturizer(atom_featurizer, bond_featurizer)
