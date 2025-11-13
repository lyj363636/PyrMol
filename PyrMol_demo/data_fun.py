import warnings
warnings.filterwarnings("ignore")
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from gensim.models import word2vec

import torch
from torch.utils.data import Dataset
import dgl
from dgl.dataloading import GraphDataLoader
from rdkit.Chem.BRICS import FindBRICSBonds, BreakBRICSBonds
from rdkit.Chem import ChemicalFeatures
from rdkit.Chem import MACCSkeys
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')
import os
import networkx as nx
######################
### Import Library ###
######################

import os as os
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
# rdkit
from rdkit import Chem
from rdkit import RDConfig
from rdkit.Chem import AllChem
from rdkit.Chem import FragmentCatalog
from rdkit.Chem import rdDepictor

rdDepictor.SetPreferCoordGen(True)


#########################
### Global Definition ###
#########################

# for junction tree and cluster
MST_MAX_WEIGHT = 100

# for every graph
definedAtom = [
    'C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg',
    'Na', 'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl',
    'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H',  # H?
    'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
    'Pt', 'Hg', 'Pb',
    'Unknown'
]
NUMBER_OF_ATOM = len(definedAtom)

definedBond = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC
]
NUMBER_OF_BOND = len(definedBond)

# for functional
fName = os.path.join(RDConfig.RDDataDir, 'FunctionalGroups.txt')
fparams = FragmentCatalog.FragCatParams(1, 6, fName)
NUMBER_OF_FUNCTIONAL = len(range(fparams.GetNumFuncGroups()))

definedFuncBond = [
    ('C', Chem.rdchem.BondType.SINGLE, 'C'),
    ('C', Chem.rdchem.BondType.DOUBLE, 'C'),
    ('C', Chem.rdchem.BondType.TRIPLE, 'C'),
    ('C', Chem.rdchem.BondType.SINGLE, 'O'),
    ('C', Chem.rdchem.BondType.DOUBLE, 'O'),
    ('C', Chem.rdchem.BondType.SINGLE, 'N'),
    ('C', Chem.rdchem.BondType.DOUBLE, 'N'),
    ('C', Chem.rdchem.BondType.TRIPLE, 'N'),
    ('C', Chem.rdchem.BondType.SINGLE, 'S'),
    ('C', Chem.rdchem.BondType.DOUBLE, 'S'),
    ('C', Chem.rdchem.BondType.TRIPLE, 'S'),
    ('O', Chem.rdchem.BondType.SINGLE, 'O'),
    ('O', Chem.rdchem.BondType.SINGLE, 'N'),
    ('O', Chem.rdchem.BondType.DOUBLE, 'N'),
    ('O', Chem.rdchem.BondType.SINGLE, 'S'),
    ('N', Chem.rdchem.BondType.SINGLE, 'N'),
    ('N', Chem.rdchem.BondType.DOUBLE, 'N'),
    ('N', Chem.rdchem.BondType.SINGLE, 'S'),
    ('N', Chem.rdchem.BondType.TRIPLE, 'S'),
    ('N', Chem.rdchem.BondType.DOUBLE, 'S'),
    ('S', Chem.rdchem.BondType.SINGLE, 'S'),
    ('S', Chem.rdchem.BondType.DOUBLE, 'S'),
    ('S', Chem.rdchem.BondType.TRIPLE, 'S'),
    'Unknown'
]
NUMBER_OF_FUNCBOND = len(definedFuncBond)

fring = pd.read_csv('util/func_ring.txt', sep='\t')
definedRing = list(fring['smarts'])
NUMBER_OF_FUNCRING = len(definedRing)


##########################
### Utilities Function ###
##########################

def one_of_k_encoding(x, allowable_set):
    # Maps inputs only in the allowable set
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set {1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


# one-hot encoding (unknown to last element)
def one_of_k_encoding_unk(x, allowable_set):
    # Maps inputs not in the allowable set to the last element
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


# one-hot encoding (unknown all zeros)
def one_of_k_encoding_none(x, allowable_set):
    # Maps inputs not in the allowable set to zero list
    if x not in allowable_set:
        x = [0 for i in range(len(allowable_set))]
    return list(map(lambda s: x == s, allowable_set))


# print global definition
def printGlobalDefinition():
    print('ATOM', NUMBER_OF_ATOM)
    for (i, item) in enumerate(definedAtom):
        print(i, item)
    print('BOND', NUMBER_OF_BOND)
    for (i, item) in enumerate(definedBond):
        print(i, item)
    print('FUNCTIONAL', NUMBER_OF_FUNCTIONAL)
    for i in range(fparams.GetNumFuncGroups()):
        if i == 27:
            print('SWAP WITH 29')
        if i == 29:
            print('SWAP WITH 27')
        print(i, fparams.GetFuncGroup(i).GetProp('_Name'), Chem.MolToSmarts(fparams.GetFuncGroup(i)))
    print('FUNC_BOND', NUMBER_OF_FUNCBOND)
    for (i, item) in enumerate(definedFuncBond):
        print(i, item)
    print('FUNC_RING', NUMBER_OF_FUNCRING)
    print(fring)


############################
### Class Initialization ###
############################

# MoleculeGraph

fdefName = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
factory = ChemicalFeatures.BuildFeatureFactory(fdefName)


def bond_features(bond: Chem.rdchem.Bond):
    if bond is None:
        # fbond = [1] + [0] * (BOND_FDIM - 1)
        print("this bond is None")
    else:
        bt = bond.GetBondType()
        fbond = [
            0,  # bond is not None
            bt == Chem.rdchem.BondType.SINGLE,
            bt == Chem.rdchem.BondType.DOUBLE,
            bt == Chem.rdchem.BondType.TRIPLE,
            bt == Chem.rdchem.BondType.AROMATIC,
            (bond.GetIsConjugated() if bt is not None else 0),
            (bond.IsInRing() if bt is not None else 0)
        ]
        fbond += onek_encoding_unk(int(bond.GetStereo()), list(range(6)))

    return fbond


def pharm_property_types_feats(mol, factory=factory):
    types = [i.split('.')[1] for i in factory.GetFeatureDefs().keys()]
    feats = [i.GetType() for i in factory.GetFeaturesForMol(mol)]
    result = [0] * len(types)
    for i in range(len(types)):
        if types[i] in list(set(feats)):
            result[i] = 1
    return result


def GetBricsBonds(mol):
    brics_bonds = list()
    brics_bonds_rules = list()
    bonds_tmp = FindBRICSBonds(mol)
    bonds = [b for b in bonds_tmp]
    for item in bonds:  # item[0] is bond, item[1] is brics type
        brics_bonds.append([int(item[0][0]), int(item[0][1])])
        brics_bonds_rules.append([[int(item[0][0]), int(item[0][1])], GetBricsBondFeature([item[1][0], item[1][1]])])
        brics_bonds.append([int(item[0][1]), int(item[0][0])])
        brics_bonds_rules.append([[int(item[0][1]), int(item[0][0])], GetBricsBondFeature([item[1][1], item[1][0]])])
    result = []
    for bond in mol.GetBonds():
        beginatom = bond.GetBeginAtomIdx()
        endatom = bond.GetEndAtomIdx()
        if [beginatom, endatom] in brics_bonds:
            result.append([bond.GetIdx(), beginatom, endatom])
    return result, brics_bonds_rules


def GetBricsBondFeature(action):
    result = []
    start_action_bond = int(action[0]) if (action[0] != '7a' and action[0] != '7b') else 7
    end_action_bond = int(action[1]) if (action[1] != '7a' and action[1] != '7b') else 7
    emb_0 = [0 for i in range(17)]
    emb_1 = [0 for i in range(17)]
    emb_0[start_action_bond] = 1
    emb_1[end_action_bond] = 1
    result = emb_0 + emb_1
    return result


def maccskeys_emb(mol):
    return list(MACCSkeys.GenMACCSKeys(mol))


def mol_with_atom_index(mol):
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx() + 1)  # aviod index 0
    return mol


def GetMolecularFeats(mol):
    emb_0 = maccskeys_emb(mol)  # MaccsKey指纹，二进制长度
    emb_1 = pharm_property_types_feats(mol)
    emb = emb_0 + emb_1
    return emb


ELEMENTS = [35, 6, 7, 8, 9, 15, 16, 17, 53]

ATOM_FEATURES = {
    'atomic_num': ELEMENTS,
    'degree': [0, 1, 2, 3, 4, 5],
    'formal_charge': [-1, -2, 1, 2, 0],
    'chiral_tag': [0, 1, 2, 3],
    'num_Hs': [0, 1, 2, 3, 4],
    'hybridization': [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2
    ],
}


def onek_encoding_unk(value, choices):
    encoding = [0] * (len(choices) + 1)
    index = choices.index(value) if value in choices else -1
    encoding[index] = 1

    return encoding


def atom_features(atom: Chem.rdchem.Atom, dim=None):
    features = onek_encoding_unk(atom.GetAtomicNum(), ATOM_FEATURES['atomic_num']) + \
               onek_encoding_unk(atom.GetTotalDegree(), ATOM_FEATURES['degree']) + \
               onek_encoding_unk(atom.GetFormalCharge(), ATOM_FEATURES['formal_charge']) + \
               onek_encoding_unk(int(atom.GetChiralTag()), ATOM_FEATURES['chiral_tag']) + \
               onek_encoding_unk(int(atom.GetTotalNumHs()), ATOM_FEATURES['num_Hs']) + \
               onek_encoding_unk(int(atom.GetHybridization()), ATOM_FEATURES['hybridization']) + \
               [1 if atom.GetIsAromatic() else 0] + \
               [atom.GetMass() * 0.01]  # scaled to about the same range as other features
    if dim:
        if dim > len(features):
            return features + [0 for i in range(dim - len(features))]
    else:
        return features


two_edge_dict = {
    "a": "aba",  # 原子节点关系
    "s": "sbs",  # 子结构关系
    "f": "fbf",  # 官能团关系
    "p": "pbp",  # 药效团关系
    "j": "jbj",  # 链接树关系
    "m": "mbm",  # 链接树关系

    "as": 'ajs',  # 原子到子结构关系
    "af": 'ajf',  # 原子到官能团关系
    "ap": 'ajp',  # 原子到药效团关系
    "aj": 'ajj',  # 原子到链接树关系

    "sm": "sjm",  # 原子到子结构关系
    "fm": "fjm",  # 官能团到分子
    "pm": "pjm",  # 药效团到分子
    "jm": "jjm"  # 链接树到分子
}

multi_edge_dict = {
    "a": ("a", "aba", "a"),  # 原子节点关系
    "s": ("a", "sbs", "a"),  # 子结构关系
    "f": ("f", "fbf", "f"),  # 官能团关系
    "p": ("p", "pbp", "p"),  # 药效团关系
    "j": ("j", "jbj", "j"),  # 链接树关系
    "m": ("m", "mbm", "m"),  # 链接树关系

    "as": ("a", "ajs", "s"),  # 原子到子结构关系
    "af": ("a", "ajf", "f"),  # 原子到官能团关系
    "ap": ("a", "ajp", "p"),  # 原子到药效团关系
    "aj": ("a", "ajj", "j"),  # 原子到链接树关系

    "sm": ("s", "sjm", "m"),  # 原子到子结构关系
    "fm": ("f", "fjm", "m"),  # 官能团到分子
    "pm": ("p", "pjm", "m"),  # 药效团到分子
    "jm": ("j", "jjm", "m")  # 链接树到分子
}


def get_edge_types(node_types, mold="two"):
    node_types = [i for i in node_types]
    if mold == "two":
        edge_types = ["aba", "mbm"]
        for node in node_types:
            edge_types.append(two_edge_dict[f'{node}'])
            edge_types.append(two_edge_dict[f'a{node}'])
            edge_types.append(two_edge_dict[f'{node}m'])
    elif mold == "multi":
        edge_types = [("a", "aba", "a"), ("m", "mbm", "m")]
        for node in node_types:
            edge_types.append(multi_edge_dict[f'{node}'])
            edge_types.append(multi_edge_dict[f'a{node}'])
            edge_types.append(multi_edge_dict[f'{node}m'])
    return edge_types


def is_same_atom(a1, a2):
    return a1['symbol'] == a2['symbol']


def is_same_bond(b1, b2):
    return b1['type'] == b2['type']


def is_isomorphic(topology1, topology2):
    return nx.is_isomorphic(topology1, topology2, node_match=is_same_atom, edge_match=is_same_bond)


# check graph is isomorphic only atom
def is_isomorphic_atom(topology1, topology2):
    return nx.is_isomorphic(topology1, topology2, node_match=is_same_atom)


# convert mol to topology (atom symbol, no consider bond type)
def topology_checker(mol):
    topology = nx.Graph()
    for atom in mol.GetAtoms():
        # Add the atoms as nodes
        topology.add_node(atom.GetIdx(), symbol=atom.GetSymbol())
    for bond in mol.GetBonds():
        # Add the bonds as edges
        topology.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), type=bond.GetBondType())

    return topology


def substruct2mol(mol, clique):
    sub_mol = Chem.RWMol(Chem.Mol())

    # 添加原子到子结构中
    for idx in clique:
        atom = mol.GetAtomWithIdx(idx)
        sub_mol.AddAtom(atom)

    # 添加键到子结构中
    for bond in mol.GetBonds():
        begin_atom_idx = bond.GetBeginAtomIdx()
        end_atom_idx = bond.GetEndAtomIdx()
        if begin_atom_idx in clique and end_atom_idx in clique:
            sub_mol.AddBond(clique.index(begin_atom_idx), clique.index(end_atom_idx), bond.GetBondType())

    # 转换为分子对象
    sub_mol = sub_mol.GetMol()
    return sub_mol


class MolSentence:
    """Class for storing mol sentences in pandas DataFrame
    """

    def __init__(self, sentence):
        self.sentence = sentence
        if type(self.sentence[0]) != str:
            raise TypeError('List with strings expected')

    def __len__(self):
        return len(self.sentence)

    def __str__(self):  # String representation
        return 'MolSentence with %i words' % len(self.sentence)

    __repr__ = __str__  # Default representation

    def contains(self, word):
        """Contains (and __contains__) method enables usage of "'Word' in MolSentence"""
        if word in self.sentence:
            return True
        else:
            return False

    __contains__ = contains  # MolSentence.contains('word')

    def __iter__(self):  # Iterate over words (for word in MolSentence:...)
        for x in self.sentence:
            yield x

    _repr_html_ = __str__


def mol2alt_sentence(mol, radius):
    """Same as mol2sentence() expect it only returns the alternating sentence
    Calculates ECFP (Morgan fingerprint) and returns identifiers of substructures as 'sentence' (string).
    Returns a tuple with 1) a list with sentence for each radius and 2) a sentence with identifiers from all radii
    combined.
    NOTE: Words are ALWAYS reordered according to atom order in the input mol object.
    NOTE: Due to the way how Morgan FPs are generated, number of identifiers at each radius is smaller

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
    radius : float
        Fingerprint radius

    Returns
    -------
    list
        alternating sentence
    combined
    """
    radii = list(range(int(radius) + 1))
    info = {}
    _ = AllChem.GetMorganFingerprint(mol, radius, bitInfo=info)  # info: dictionary identifier, atom_idx, radius

    mol_atoms = [a.GetIdx() for a in mol.GetAtoms()]
    dict_atoms = {x: {r: None for r in radii} for x in mol_atoms}

    for element in info:
        for atom_idx, radius_at in info[element]:
            dict_atoms[atom_idx][radius_at] = element  # {atom number: {fp radius: identifier}}


    identifiers_alt = []
    for atom in dict_atoms:  # iterate over atoms
        for r in radii:  # iterate over radii
            identifiers_alt.append(dict_atoms[atom][r])

    alternating_sentence = map(str, [x for x in identifiers_alt if x])

    return list(alternating_sentence)


def get_main_mol(smiles):
    smiles = smiles.split(".")
    mols = [Chem.MolFromSmiles(_) for _ in smiles]
    atom_nums = [mol.GetNumAtoms() for mol in mols]
    max_mol = mols[atom_nums.index(max(atom_nums))]
    return max_mol


def Atom2Substructure(mol, radius, model, subkeys, unseen_vec):
    """为每一个原子节点生成包含改节点的子结构列表
    每一个原子对应1个或者多个子结构
    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
    radius : float
        Fingerprint radius

    Returns
    -------
    list，第i个表示包含第i个原子的子结构列表
    """
    radii = list(range(int(radius) + 1))
    info = {}
    _ = AllChem.GetMorganFingerprint(mol, radius, bitInfo=info)  # info: dictionary identifier, atom_idx, radius

    mol_atoms = [a.GetIdx() for a in mol.GetAtoms()]
    dict_atoms = {x: {r: None for r in radii} for x in mol_atoms}

    for element in info:
        for atom_idx, radius_at in info[element]:
            dict_atoms[atom_idx][radius_at] = element  # {atom number: {fp radius: identifier}}

    # merge identifiers alternating radius to sentence: atom 0 radius0, atom 0 radius 1, etc.
    atom2substructions = []
    for atom in dict_atoms:  # iterate over atoms
        substructions = []
        for r in radii:  # iterate over radii
            sub_num = str(dict_atoms[atom][r])
            if sub_num in subkeys:
                substructions.append(model.wv.word_vec(sub_num))
            else:
                substructions.append(unseen_vec)
        atom2substructions.append(np.mean(np.array(substructions), 0))
    return np.array(atom2substructions)


edge_dim_dict = {
    "s": 34,
    "j": 6,
    "p": 3,
    "f": 20,
}


def edge_f_init(e_f, subg):
    edge_f = np.zeros((len(e_f), 63))
    if subg == "s":
        edge_f[:, :34] = e_f
    elif subg == "j":
        edge_f[:, 34:40] = e_f
    elif subg == "p":
        edge_f[:, 40:43] = e_f
    elif subg == "f":
        edge_f[:, 43:63] = e_f
    return edge_f.tolist()


edge_dict = {
    "a": ("a", "b", "a"),  # 原子节点关系 bind
    "s": ("s", "c", "s"),  # 子结构关系 connect
    "m": ("m", "i", "m"),  # 分子到分析 identity
    "as": ("a", "j", "s"),  # 原子到子结构关系 junction
    "sm": ("s", "d", "m"),  # 子结构到分子关系 donate
}
"""
atom dim:42
s dim:327
j dim:383
p dim:306
f dim:415
m dim:300
aa dim:14
ss dim:34
jj dim:6
pp dim:3
ff dim:20
"""

from tqdm import tqdm


class MolGraphSet_our(Dataset):
    def __init__(self, df, target, protrain_model="Mol2Vec", log=print, node_types="sjpf"):
        # self.data = df.head(100)
        self.data = df
        self.smiles = []
        self.mols = []
        self.labels = []
        self.graphs = []
        self.node_types = [i for i in node_types]
        self.edge_types = {
            "a": ("a", "b", "a"),  # 原子节点关系 bind
            "s": ("s", "c", "s"),  # 子结构关系 connect
            # "m":("m","i","m"),  # 分子到分析 identity
            "as": ("a", "j", "s"),  # 原子到子结构关系 junction
            "sm": ("s", "d", "m"),  # 子结构到分子关系 donate
        }
        self.subg_dim = 531  # 300+27+83+6+115
        self.edge_dim = 63  # 14+34+6+3+20
        if protrain_model == "Mol2Vec":
            self.Mol2Vec = word2vec.Word2Vec.load('./model_300dim.pkl')
            try:
                self.keys = set(self.Mol2Vec.wv.key_to_index.keys())
            except:
                self.keys = set(self.Mol2Vec.wv.vocab.keys())
            self.unseen = 'UNK'
            self.unseen_vec = self.Mol2Vec.wv.word_vec(self.unseen)
        error_smiles = []
        for i, row in tqdm(self.data.iterrows(), total=self.data.shape[0]):
            smiles = row['smiles']
            label = row[target].values.astype(float)
            try:
                mol = Chem.MolFromSmiles(smiles)
                atom_num = mol.GetNumAtoms()
                if mol != None:
                    if atom_num >= 3:
                        self.matrix = Atom2Substructure(mol, 1, self.Mol2Vec, self.keys, self.unseen_vec)
                        try:
                            g = self.Mol2HeteroGraph(mol)
                            self.smiles.append(smiles)
                            self.mols.append(mol)
                            self.graphs.append(g)
                            self.labels.append(label)
                        except:
                            error_smiles.append(smiles)
                    else:
                        continue
                        # log(f"{smiles} too small")
                else:
                    error_smiles.append(smiles)
            except:
                error_smiles.append(smiles)
        print(len(error_smiles) / self.data.shape[0])
        # except Exception as e:
        #     log(e, 'invalid', smiles)

    def __len__(self):
        return len(self.mols)

    def __getitem__(self, idx):

        return self.smiles[idx], self.graphs[idx], self.labels[idx]

    def Mol2HeteroGraph(self, mol):

        edges = {edge_type: [] for node_type, edge_type in self.edge_types.items()}
        node_features = {node_type: [] for node_type in ["a", "s", "m"]}
        edge_features = {edge_type: [] for node_type, edge_type in self.edge_types.items()}
        # """构建分子图"""
        # edges[self.edge_types["m"]].append([0, 0])
        """构建原子图"""
        for bond in mol.GetBonds():
            edges[self.edge_types["a"]].append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
            edges[self.edge_types["a"]].append([bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()])
        """生成原子节点和边特征"""
        f_atom = []
        for atom in mol.GetAtoms():
            # f_atom.append(atom_features(atom))
            f_atom.append(np.concatenate([self.matrix[atom.GetIdx()], np.array(atom_features(atom))]))
        node_features["a"] = f_atom
        a_bond_f = []
        for src, dst in edges[self.edge_types["a"]]:
            a_bond_f.append(bond_features(mol.GetBondBetweenAtoms(src, dst)))
        edge_features[self.edge_types["a"]] = a_bond_f  # dim=14
        sub_num = 0
        for node_type in self.node_types:
            if node_type == 's':
                """构建子结构图"""
                result_ap, result_p = self.GetFragmentFeats(mol)
                reac_idx, bbr = GetBricsBonds(mol)
                for r in reac_idx:
                    begin = r[1]
                    end = r[2]
                    edges[self.edge_types["s"]].append([result_ap[begin] + sub_num, result_ap[end] + sub_num])
                    edges[self.edge_types["s"]].append([result_ap[end] + sub_num, result_ap[begin] + sub_num])
                for k, v in result_ap.items():
                    edges[self.edge_types["as"]].append([k, v + sub_num])
                for v in set(result_ap.values()):
                    edges[self.edge_types["sm"]].append([v + sub_num, 0])
                sub_num += len(set(result_ap.values()))

                """生成子结构节点和边特征"""
                f_substruct = []
                for k, v in result_p.items():
                    f_substruct.append(v)
                n_feature = np.zeros((len(f_substruct), self.subg_dim))
                n_feature[:, :327] = f_substruct
                node_features["s"].extend(n_feature.tolist())
                s_bond_f = []
                if len(edges[self.edge_types["s"]]) > 1:
                    for src, dst in edges[self.edge_types["s"]]:
                        p0_g = src
                        p1_g = dst
                        for i in bbr:
                            p0 = result_ap[i[0][0]]
                            p1 = result_ap[i[0][1]]
                            if p0_g == p0 and p1_g == p1:
                                s_bond_f.append(i[1])
                    edge_feature = edge_f_init(s_bond_f, node_type)
                    edge_features[self.edge_types["s"]].extend(edge_feature)  # dim=34

            elif node_type == 'f':
                """构建官能团图"""
                cliques, f_edges, cliques_func, cliques_ring = self.getFunctionalGraph(mol)
                f_set = set()
                for r in f_edges:
                    edges[self.edge_types["s"]].append([r[0] + sub_num, r[1] + sub_num])
                    f_set.add(r[0] + sub_num)
                    f_set.add(r[1] + sub_num)
                for f_index, b_list in enumerate(cliques):
                    for b in b_list:
                        edges[self.edge_types["as"]].append([b, f_index + sub_num])
                for f in f_set:
                    edges[self.edge_types["sm"]].append([f, 0])
                sub_num += len(f_set)
                """生成官能团节点和边特征"""
                clique_size, clique_index, clique_attr, cliqueedge_index, cliqueedge_attr = self.mol_to_Functional_graph(
                    mol,
                    cliques,
                    f_edges,
                    cliques_func,
                    cliques_ring,
                    True)
                n_feature = np.zeros((len(clique_attr), self.subg_dim))
                # n_feature[:, :415] = clique_attr #300+27+83+6+115
                n_feature[:, :300] = np.array(clique_attr)[:, :300]  # 300+27+83+6+115
                n_feature[:, 416:] = np.array(clique_attr)[:, 300:]  # 300+27+83+6+115
                node_features["s"].extend(n_feature.tolist())
                edge_feature = edge_f_init(cliqueedge_attr, node_type)
                edge_features[self.edge_types["s"]].extend(edge_feature)  # dim = 20

            elif node_type == 'p':
                """构建药效团图"""
                cliques, p_edges, cliques_prop = self.getPharmacophoreGraph(mol)
                p_set = set()
                for r in p_edges:
                    edges[self.edge_types["s"]].append([r[0] + sub_num, r[1] + sub_num])
                    p_set.add(r[0] + sub_num)
                    p_set.add(r[1] + sub_num)
                atoms_num = mol.GetNumAtoms()
                cliques_ = [item for sublist in cliques for item in sublist]
                if max(cliques_) != (atoms_num - 1) or min(cliques_) != 0:
                    print(atoms_num, cliques)

                for p_index, b_list in enumerate(cliques):
                    for b in b_list:
                        edges[self.edge_types["as"]].append([b, p_index + sub_num])
                for p in p_set:
                    edges[self.edge_types["sm"]].append([p, 0])
                sub_num += len(p_set)
                clique_size, clique_index, clique_attr, cliqueedge_index, cliqueedge_attr = self.mol_to_Pharmacophore_graph(
                    mol, cliques, p_edges, cliques_prop)
                n_feature = np.zeros((len(clique_attr), self.subg_dim))
                # n_feature[:, :306] = clique_attr
                n_feature[:, :300] = np.array(clique_attr)[:, :300]  # 300+27+83+6+115
                n_feature[:, 410:416] = np.array(clique_attr)[:, 300:]  # 300+27+83+6+115
                node_features["s"].extend(n_feature.tolist())
                edge_feature = edge_f_init(cliqueedge_attr, node_type)
                edge_features[self.edge_types["s"]].extend(edge_feature)  # dim=3

            elif node_type == 'j':
                """构建连接树图"""
                cliques, j_edges = self.getJunctionTreeGraph(mol)
                j_set = set()
                for r in j_edges:
                    edges[self.edge_types["s"]].append([r[0] + sub_num, r[1] + sub_num])
                    j_set.add(r[0] + sub_num)
                    j_set.add(r[1] + sub_num)
                for j_index, b_list in enumerate(cliques):
                    for b in b_list:
                        edges[self.edge_types["as"]].append([b, j_index + sub_num])
                for j in j_set:
                    edges[self.edge_types["sm"]].append([j, 0])
                sub_num += len(j_set)
                clique_size, clique_index, clique_attr, cliqueedge_index, cliqueedge_attr = self.mol_to_JunctionTree_graph(
                    mol,
                    cliques,
                    j_edges,
                    normalize=True)
                n_feature = np.zeros((len(clique_attr), self.subg_dim))
                # n_feature[:, :383] = clique_attr
                n_feature[:, :300] = np.array(clique_attr)[:, :300]  # 300+27+83+6+115
                n_feature[:, 327:410] = np.array(clique_attr)[:, 300:]  # 300+27+83+6+115
                node_features["s"].extend(n_feature.tolist())
                edge_feature = edge_f_init(cliqueedge_attr, node_type)
                edge_features[self.edge_types["s"]].extend(edge_feature)  # dim=6

        """构建异构图"""
        g = dgl.heterograph(edges)  # 构建异构图
        # g = dgl.add_self_loop(g, etype="b")
        # g = dgl.add_self_loop(g, etype="c")
        """生成分子节点特征"""
        # f_mol = GetMolecularFeats(mol)
        f_mol = self.matrix.mean(0)
        g.nodes['m'].data['f'] = torch.FloatTensor([f_mol])

        # self.node_types.append("a")
        node_dims = {}
        for node_type in ["a", "s"]:
            max_len = max(len(sublist) for sublist in node_features[node_type])
            # padded_data = [sublist + [0] * (max_len - len(sublist)) for sublist in node_features[node_type]]
            f_node = torch.tensor(node_features[node_type], dtype=torch.float32)
            g.nodes[node_type].data['f'] = f_node
            node_dims[node_type] = len(f_node[0])
        f_node = torch.FloatTensor(node_features["a"])
        g.nodes["a"].data['f'] = f_node
        node_dims["a"] = len(f_node[0])

        """
        "a":("a","b","a"),  # 原子节点关系 bind
            "s":("s","c","s"),  # 子结构关系 connect
            "m":("m","i","m"),  # 分子到分析 identity
            "as":("a","j","s"), # 原子到子结构关系 junction
            "sm":("s","d","m"), # 子结构到分子关系 donate"""

        return g

    def Mol2AtomGraph(self, mol):

        edges = {k: [] for k in self.edge_types}
        node_features = {}
        edge_features = {}
        """构建分子图"""
        edges[self.edge_type["m"]].append([0, 0])
        """构建原子图"""
        for bond in mol.GetBonds():
            edges[self.edge_type["a"]].append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
            edges[self.edge_type["a"]].append([bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()])
        """生成原子节点和边特征"""
        f_atom = []
        for atom in mol.GetAtoms():
            f_atom.append(atom_features(atom))
        node_features["a"] = f_atom
        a_bond_f = []
        for src, dst in edges[self.edge_type["a"]]:
            a_bond_f.append(bond_features(mol.GetBondBetweenAtoms(src, dst)))
        edge_features[self.edge_type["a"]] = a_bond_f

        for node_type in self.node_types:
            if node_type == 's':
                """构建子结构图"""
                result_ap, result_p = self.GetFragmentFeats(mol)
                reac_idx, bbr = GetBricsBonds(mol)
                for r in reac_idx:
                    begin = r[1]
                    end = r[2]
                    edges[self.edge_type["s"]].append([result_ap[begin], result_ap[end]])
                    edges[self.edge_type["s"]].append([result_ap[end], result_ap[begin]])
                for k, v in result_ap.items():
                    edges[self.edge_type["as"]].append([k, v])
                for v in set(result_ap.values()):
                    edges[self.edge_type["sm"]].append([v, 0])
                """生成子结构节点和边特征"""
                f_substruct = []
                for k, v in result_p.items():
                    f_substruct.append(v)
                node_features["s"] = f_substruct
                s_bond_f = []
                for src, dst in edges[self.edge_type["s"]]:
                    p0_g = src
                    p1_g = dst
                    for i in bbr:
                        p0 = result_ap[i[0][0]]
                        p1 = result_ap[i[0][1]]
                        if p0_g == p0 and p1_g == p1:
                            s_bond_f.append(i[1])
                edge_features[self.edge_type["s"]] = s_bond_f

            elif node_type == 'f':
                """构建官能团图"""
                cliques, f_edges, cliques_func, cliques_ring = self.getFunctionalGraph(mol)
                f_set = set()
                for r in f_edges:
                    edges[self.edge_type["f"]].append([r[0], r[1]])
                    edges[self.edge_type["f"]].append([r[1], r[0]])
                    f_set.add(r[0])
                    f_set.add(r[1])
                for f_index, b_list in enumerate(cliques):
                    for b in b_list:
                        edges[self.edge_type["af"]].append([b, f_index])
                for f in f_set:
                    edges[self.edge_type["fm"]].append([f, 0])
                """生成官能团节点和边特征"""
                clique_size, clique_index, clique_attr, cliqueedge_index, cliqueedge_attr = self.mol_to_Functional_graph(
                    mol,
                    cliques,
                    f_edges,
                    cliques_func,
                    cliques_ring,
                    True)
                node_features["f"] = clique_attr
                edge_features[self.edge_type["f"]] = cliqueedge_attr

            elif node_type == 'p':
                """构建药效团图"""
                cliques, p_edges, cliques_prop = self.getPharmacophoreGraph(mol)
                p_set = set()
                for r in p_edges:
                    edges[self.edge_type["p"]].append([r[0], r[1]])
                    edges[self.edge_type["p"]].append([r[1], r[0]])
                    p_set.add(r[0])
                    p_set.add(r[1])
                atoms_num = mol.GetNumAtoms()
                cliques_ = [item for sublist in cliques for item in sublist]
                if max(cliques_) != (atoms_num - 1) or min(cliques_) != 0:
                    print(atoms_num, cliques)

                for p_index, b_list in enumerate(cliques):
                    for b in b_list:
                        edges[self.edge_type["ap"]].append([b, p_index])
                for p in p_set:
                    edges[self.edge_type["pm"]].append([p, 0])
                clique_size, clique_index, clique_attr, cliqueedge_index, cliqueedge_attr = self.mol_to_Pharmacophore_graph(
                    mol, cliques, p_edges, cliques_prop)
                node_features["p"] = clique_attr
                edge_features[self.edge_type["p"]] = cliqueedge_attr

            elif node_type == 'j':
                """构建连接树图"""
                cliques, j_edges = self.getJunctionTreeGraph(mol)
                j_set = set()
                for r in j_edges:
                    edges[self.edge_type["j"]].append([r[0], r[1]])
                    edges[self.edge_type["j"]].append([r[1], r[0]])
                    j_set.add(r[0])
                    j_set.add(r[1])
                for j_index, b_list in enumerate(cliques):
                    for b in b_list:
                        edges[self.edge_type["aj"]].append([b, j_index])
                for j in j_set:
                    edges[self.edge_type["jm"]].append([j, 0])
                clique_size, clique_index, clique_attr, cliqueedge_index, cliqueedge_attr = self.mol_to_JunctionTree_graph(
                    mol,
                    cliques,
                    j_edges,
                    normalize=True)
                node_features["j"] = clique_attr
                edge_features[self.edge_type["j"]] = cliqueedge_attr

        """构建异构图"""
        g = dgl.heterograph(edges)  # 构建异构图
        """生成分子节点特征"""
        # f_mol = GetMolecularFeats(mol)
        f_mol = self.matrix.mean(0)
        g.nodes['m'].data['f'] = torch.FloatTensor([f_mol])

        # self.node_types.append("a")
        node_dims = {}
        for node_type in self.node_types:
            f_node = torch.tensor(node_features[node_type], dtype=torch.float32)
            g.nodes[node_type].data['f'] = f_node
            node_dims[node_type] = len(f_node[0])
        f_node = torch.FloatTensor(node_features["a"])
        g.nodes["a"].data['f'] = f_node
        node_dims["a"] = len(f_node[0])

        node_nums = {}
        for node_type in self.node_types:
            node_nums[node_type] = g.nodes[node_type].data['f'].size()[0]
        node_nums["a"] = g.nodes["a"].data['f'].size()[0]
        max_dim = max(node_dims.values())

        return g

    def GetFragmentFeats(self, mol):
        break_bonds = [mol.GetBondBetweenAtoms(i[0][0], i[0][1]).GetIdx() for i in FindBRICSBonds(mol)]
        if break_bonds == []:
            tmp = mol
        else:
            tmp = Chem.FragmentOnBonds(mol, break_bonds, addDummies=False)
        frags_idx_lst = Chem.GetMolFrags(tmp)
        result_ap = {}
        result_p = {}
        pharm_id = 0
        for frag_idx in frags_idx_lst:  # 片段
            for atom_id in frag_idx:
                result_ap[atom_id] = pharm_id
            try:
                mol_pharm = Chem.MolFromSmiles(Chem.MolFragmentToSmiles(mol, frag_idx))
                emb_0 = self.matrix[list(frag_idx)].mean(0).tolist()
                # emb_0 = maccskeys_emb(mol_pharm)  # MaccsKey指纹，二进制长度
                emb_1 = pharm_property_types_feats(mol_pharm)  # 27
            except Exception:
                emb_0 = [0 for i in range(300)]
                emb_1 = [0 for i in range(27)]
            result_p[pharm_id] = emb_0 + emb_1

            pharm_id += 1
        return result_ap, result_p

    """生成官能团图"""

    def getFunctionalGraph(self, mol):
        n_atoms = mol.GetNumAtoms()

        # functional group
        funcGroupDict = dict()
        for i in range(fparams.GetNumFuncGroups()):
            funcGroupDict[i] = list(mol.GetSubstructMatches(fparams.GetFuncGroup(i)))

        # edit #27 <-> #29
        temp = funcGroupDict[27]
        funcGroupDict[27] = funcGroupDict[29]
        funcGroupDict[29] = temp

        cliques = []
        cliques_ring = {}  # node group in ring
        cliques_func = {}  # node group in func
        seen_func = {}  # node seen in func
        group_num = 0

        # extract functional from substructure match
        for f in funcGroupDict:
            for l in funcGroupDict[f]:
                if not (all(ll in seen_func for ll in l)):
                    cliques.append(list(l))
                    for ll in l:
                        if ll in seen_func or ll in cliques_func:
                            cliques_func[ll].append(f)
                            seen_func[ll].append(group_num)
                        else:
                            cliques_func[ll] = [f]
                            seen_func[ll] = [group_num]
                group_num += 1

        # extract bond which not in functional
        for bond in mol.GetBonds():
            a1 = bond.GetBeginAtom().GetIdx()
            a2 = bond.GetEndAtom().GetIdx()
            bt = bond.GetBondType()
            if not bond.IsInRing():
                if (a1 not in seen_func) or (a2 not in seen_func):
                    cliques.append([a1, a2])
                elif a1 in seen_func and a2 in seen_func and len(set(seen_func[a1]) & set(seen_func[a2])) == 0:
                    cliques.append([a1, a2])

        # extract ring
        ssr = [list(x) for x in Chem.GetSymmSSSR(mol)]
        cliques.extend(ssr)
        cliques_ring = ssr

        nei_list = [[] for i in range(n_atoms)]
        for i in range(len(cliques)):
            for atom in cliques[i]:
                nei_list[atom].append(i)


        cliques = [c for c in cliques if len(c) > 0]
        nei_list = [[] for i in range(n_atoms)]
        for i in range(len(cliques)):
            for atom in cliques[i]:
                nei_list[atom].append(i)

        # Build edges and add singleton cliques
        edges = defaultdict(int)
        for atom in range(n_atoms):
            if len(nei_list[atom]) <= 1:
                continue
            cnei = nei_list[atom]
            # Number of bond clusters that the atom lies in.
            bonds = [c for c in cnei if len(cliques[c]) == 2]
            # Number of ring clusters that the atom lies in.
            funcring = [c for c in cnei if len(cliques[c]) > 2]



            for i in range(len(cnei)):
                for j in range(i + 1, len(cnei)):
                    c1, c2 = cnei[i], cnei[j]
                    inter = set(cliques[c1]) & set(cliques[c2])
                    if edges[(c1, c2)] < len(inter):
                        # cnei[i] < cnei[j] by construction ?
                        edges[(c1, c2)] = len(inter)
                        edges[(c2, c1)] = len(inter)

        # check isolated single atom
        for atom in mol.GetAtoms():
            if atom.GetIdx() not in nei_list[atom.GetIdx()] and len(atom.GetBonds()) == 0:
                cliques.append([atom.GetIdx()])
                nei_list[atom.GetIdx()].append(len(cliques) - 1)

        edges = [i for i in edges]

        return cliques, edges, cliques_func, cliques_ring

    def getCliqueFeatures_funcGroup(self, clique, edges, clique_idx, cliques_func, cliques_ring, mol):
        # number of node features (115)
        funcType = [0 for f in range(len(range(fparams.GetNumFuncGroups())))]  # no unknown
        funcRingTypeList = range(len(definedRing))  # (only aromatic)
        funcRingTypeOtherList = range(len(definedRing))  # (other bonds)
        funcRingTypeSizeList = [3, 4, 5, 6, 7, 8, 9, 10]  # unknown ring size 3-9 and >9
        funcBondTypeList = range(len(definedFuncBond))  # included unknown
        # atomTypeList = range(len(definedAtom)) # included unknown

        ringType = one_of_k_encoding_none(None, funcRingTypeList)
        ringTypeOther = one_of_k_encoding_none(None, funcRingTypeOtherList)
        ringTypeSize = one_of_k_encoding_none(None, funcRingTypeSizeList)
        funcBondType = one_of_k_encoding_none(None, funcBondTypeList)
        # atomType = atomType = one_of_k_encoding_none(None, atomTypeList)

        # functional group
        func_found = False
        if all(c in cliques_func for c in clique):
            funcGroup = [cliques_func[c] for c in clique]
            intersect = funcGroup[0]
            for f in funcGroup:
                intersect = set(set(intersect) & set(f))
            for i in list(intersect):
                funcType[i] = 1
                func_found = True
            if func_found:  # func, not ring not bond
                ringType = one_of_k_encoding_none(None, funcRingTypeList)
                ringTypeOther = one_of_k_encoding_none(None, funcRingTypeOtherList)
                ringTypeSize = one_of_k_encoding_none(None, funcRingTypeSizeList)
                funcBondType = one_of_k_encoding_none(None, funcBondTypeList)
        # ring type
        if len(clique) > 2 and not func_found:
            if clique in cliques_ring:
                new_mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))  # not kekulize
                smarts = Chem.MolFragmentToSmarts(new_mol, clique)
                ring_found = False
                for ring in definedRing:
                    mol_ring = Chem.MolFromSmarts(ring)
                    mol_smart = Chem.MolFromSmarts(smarts)
                    t1 = topology_checker(mol_ring)
                    t2 = topology_checker(mol_smart)
                    if len(mol_smart.GetSubstructMatches(mol_ring)) != 0:
                        ringType = one_of_k_encoding_unk(definedRing.index(ring), funcRingTypeList)
                        ringTypeOther = one_of_k_encoding_none(None, funcRingTypeOtherList)
                        ringTypeSize = one_of_k_encoding_none(None, funcRingTypeSizeList)
                        funcBondType = one_of_k_encoding_none(None, funcBondTypeList)
                        ring_found = True
                        break
                    elif is_isomorphic(t1, t2):
                        ringType = one_of_k_encoding_unk(definedRing.index(ring), funcRingTypeList)
                        ringTypeOther = one_of_k_encoding_none(None, funcRingTypeOtherList)
                        ringTypeSize = one_of_k_encoding_none(None, funcRingTypeSizeList)
                        funcBondType = one_of_k_encoding_none(None, funcBondTypeList)
                        ring_found = True
                        break
                    elif is_isomorphic_atom(t1, t2):
                        ringType = one_of_k_encoding_none(None, funcRingTypeList)
                        ringTypeOther = one_of_k_encoding_unk(definedRing.index(ring), funcRingTypeOtherList)
                        ringTypeSize = one_of_k_encoding_none(None, funcRingTypeSizeList)
                        funcBondType = one_of_k_encoding_none(None, funcBondTypeList)
                        ring_found = True
                        break
                if not ring_found:  # unknown ring
                    mol_smart = Chem.MolFromSmarts(smarts)
                    ringType = one_of_k_encoding_none(None, funcRingTypeList)
                    ringTypeOther = one_of_k_encoding_none(None, funcRingTypeOtherList)
                    ringTypeSize = one_of_k_encoding_unk(mol_smart.GetNumAtoms(), funcRingTypeSizeList)
                    funcBondType = one_of_k_encoding_none(None, funcBondTypeList)
            else:  # not ring not bond
                ringType = one_of_k_encoding_none(None, funcRingTypeList)
                ringTypeOther = one_of_k_encoding_none(None, funcRingTypeOtherList)
                ringTypeSize = one_of_k_encoding_none(None, funcRingTypeSizeList)
                funcBondType = one_of_k_encoding_none(None, funcBondTypeList)
        # bond type
        if len(clique) == 2 and not func_found:
            bond_found = False
            for bond in mol.GetBonds():
                b = bond.GetBondType()
                atom1 = bond.GetBeginAtom()
                atom2 = bond.GetEndAtom()
                a1 = atom1.GetIdx()
                a2 = atom2.GetIdx()
                a1_s = atom1.GetSymbol()
                a2_s = atom2.GetSymbol()
                if [a1, a2] == clique or [a2, a1] == clique:
                    if (a1_s, b, a2_s) in definedFuncBond:
                        funcBondType = one_of_k_encoding_unk(definedFuncBond.index((a1_s, b, a2_s)), funcBondTypeList)
                        ringType = one_of_k_encoding_none(None, funcRingTypeList)
                        ringTypeOther = one_of_k_encoding_none(None, funcRingTypeOtherList)
                        ringTypeSize = one_of_k_encoding_none(None, funcRingTypeSizeList)
                        bond_found = True
                        break
                    elif (a2_s, b, a1_s) in definedFuncBond:
                        funcBondType = one_of_k_encoding_unk(definedFuncBond.index((a2_s, b, a1_s)), funcBondTypeList)
                        ringType = one_of_k_encoding_none(None, funcRingTypeList)
                        ringTypeOther = one_of_k_encoding_none(None, funcRingTypeOtherList)
                        ringTypeSize = one_of_k_encoding_none(None, funcRingTypeSizeList)
                        bond_found = True
                        break
            if not bond_found:  # unknown bond
                ringType = one_of_k_encoding_none(None, funcRingTypeList)
                ringTypeOther = one_of_k_encoding_none(None, funcRingTypeOtherList)
                ringTypeSize = one_of_k_encoding_none(None, funcRingTypeSizeList)
                funcBondType = one_of_k_encoding_unk(None, funcBondTypeList)


        return np.array(funcType + ringType + ringTypeOther + ringTypeSize + funcBondType)

    def getCliqueEdgeFeatures_funcGroup(self, edge, clique, edge_idx, cliques_func, cliques_ring, mol):
        # number of edge features (10)+10 = (20) # (10)+12 = (22)
        begin = clique[edge[0]]
        end = clique[edge[1]]

        if all(c in cliques_func for c in begin):
            begin_type = 'func'
        elif len(begin) > 2:
            begin_type = 'ring'
        elif len(begin) == 1:
            begin_type = 'atom'
        else:
            begin_type = 'bond'

        if all(c in cliques_func for c in end):
            end_type = 'func'
        elif len(end) > 2:
            end_type = 'ring'
        elif len(end) == 1:
            end_type = 'atom'
        else:
            end_type = 'bond'

        intersect = len(set(begin) & set(end))

        begin_atom = 0
        end_atom = 0
        if intersect == 1:
            a1 = list(set(begin) & set(end))[0]
            a2 = list(set(begin) & set(end))[0]
            begin_atom = sorted(begin).index(a1) + 1
            end_atom = sorted(end).index(a2) + 1
        # in case, more than 2 common atoms
        else:
            begin_atom = 0
            end_atom = 0

        definedEdgeType = {('atom', 'atom'): 0,
                           ('atom', 'bond'): 1,
                           ('atom', 'ring'): 2,
                           ('atom', 'func'): 3,
                           ('bond', 'atom'): 1,
                           ('bond', 'bond'): 4,
                           ('bond', 'ring'): 5,
                           ('bond', 'func'): 6,
                           ('ring', 'atom'): 2,
                           ('ring', 'bond'): 5,
                           ('ring', 'ring'): 7,
                           ('ring', 'func'): 8,
                           ('func', 'atom'): 3,
                           ('func', 'bond'): 6,
                           ('func', 'ring'): 8,
                           ('func', 'func'): 9}
        edgeType = definedEdgeType[(begin_type, end_type)]

        # return np.array(one_of_k_encoding(edgeType,list(set(definedEdgeType.values())))+
        #                 one_of_k_encoding_unk(intersect,list(range(10)))+[begin_atom]+[end_atom])
        return np.array(one_of_k_encoding(edgeType, list(set(definedEdgeType.values()))) +
                        one_of_k_encoding_unk(intersect, list(range(10))))

    def mol_to_Functional_graph(self, mol, cliques, edges, cliques_func, cliques_ring, normalize=True):
        clique_size = len(cliques)

        clique_index = list(range(clique_size))
        clique_attr = []
        for idx in range(len(cliques)):
            emb_0 = self.matrix[cliques[idx]].mean(0).tolist()
            emb_1 = self.getCliqueFeatures_funcGroup(cliques[idx], edges, idx, cliques_func, cliques_ring,
                                                     mol)
            if normalize and sum(emb_1) != 0:
                clique_attr.append(emb_0 + list(emb_1 / sum(emb_1)))
            else:
                clique_attr.append(emb_0 + emb_1.tolist())

        cliqueedge_index = edges
        cliqueedge_attr = []
        for idx in range(len(edges)):
            cliqueedge_features = self.getCliqueEdgeFeatures_funcGroup(edges[idx], cliques, idx, cliques_func,
                                                                       cliques_ring, mol)
            if normalize and sum(cliqueedge_features) != 0:
                cliqueedge_attr.append(list(cliqueedge_features) / sum(cliqueedge_features))
            else:
                cliqueedge_attr.append(cliqueedge_features)

        return clique_size, clique_index, clique_attr, cliqueedge_index, cliqueedge_attr

    """生成药效团图"""

    def getPharmacophoreGraph(self, mol):
        mol = mol_with_atom_index(mol)
        mol_g = Chem.rdReducedGraphs.GenerateMolExtendedReducedGraph(mol)
        mol_g.UpdatePropertyCache(False)
        mapping_atom = {a.GetAtomMapNum(): i for i, a in enumerate(mol_g.GetAtoms())}

        cliques = []
        cliques_prop = []

        ring_8 = [list(x) for x in Chem.GetSymmSSSR(mol) if len(list(x)) < 8]
        ring_B8 = [list(x) for x in Chem.GetSymmSSSR(mol) if len(list(x)) >= 8]

        # add more 8-atom ring
        if len(ring_B8) > 0:
            rwmol_g = Chem.RWMol(mol_g)
            for rb8 in ring_B8:
                new_a = rwmol_g.AddAtom(Chem.Atom(0))
                rwmol_g.GetAtomWithIdx(new_a).SetProp('_ErGAtomTypes', '')
                for rb8_a in rb8:
                    if rb8_a in mapping_atom:
                        rwmol_g.AddBond(new_a, mapping_atom[rb8_a], Chem.BondType.SINGLE)
            mol_g = rwmol_g

        # display(mol_g)
        num_ring_8 = 0
        num_ring_B8 = 0

        for atom in mol_g.GetAtoms():
            if atom.GetAtomMapNum() == 0 and atom.GetSymbol() == '*' and 5 in list(
                    atom.GetPropsAsDict()['_ErGAtomTypes']):
                cliques.append(ring_8[num_ring_8])
                cliques_prop.append(list(atom.GetPropsAsDict()['_ErGAtomTypes']))
                num_ring_8 += 1
            elif atom.GetAtomMapNum() == 0 and atom.GetSymbol() == '*' and 4 in list(
                    atom.GetPropsAsDict()['_ErGAtomTypes']):
                cliques.append(ring_8[num_ring_8])
                cliques_prop.append(list(atom.GetPropsAsDict()['_ErGAtomTypes']))
                num_ring_8 += 1
            elif atom.GetAtomMapNum() == 0 and atom.GetSymbol() == '*':
                cliques.append(ring_B8[num_ring_B8])
                cliques_prop.append(list(atom.GetPropsAsDict()['_ErGAtomTypes']))
                num_ring_B8 += 1
            else:
                cliques.append([atom.GetAtomMapNum() - 1])
                cliques_prop.append(list(atom.GetPropsAsDict()['_ErGAtomTypes']))

        edges = []
        for bond in mol_g.GetBonds():
            a1 = bond.GetBeginAtomIdx()
            a2 = bond.GetEndAtomIdx()
            edges.append((a1, a2))
            edges.append((a2, a1))

        cliques_ = []
        for clique in cliques:
            cliques_.append([atom_index for atom_index in clique])

        return cliques_, edges, cliques_prop

    def getCliqueFeatures_pharmacophore(self, clique, edges, clique_idx, cliques_prop, mol):
        # number of node features (6)
        pharmacophore = np.zeros(6)
        for p in cliques_prop:
            pharmacophore[p] = 1

        return np.array(pharmacophore)

    def getCliqueEdgeFeatures_pharmacophore(self, edge, clique, edge_idx, cliques_prop, mol):
        # number of edge features (3)
        begin = cliques_prop[edge[0]]
        end = cliques_prop[edge[1]]

        if len(begin) == 0:
            begin_type = 'none'
        else:
            begin_type = 'phar'

        if len(end) == 0:
            end_type = 'none'
        else:
            end_type = 'phar'

        definedEdgeType = {('none', 'none'): 0,
                           ('none', 'phar'): 1,
                           ('phar', 'none'): 1,
                           ('phar', 'phar'): 2, }
        EdgeType = definedEdgeType[(begin_type, end_type)]

        return np.array(one_of_k_encoding(EdgeType, list(set(definedEdgeType.values()))))

    def mol_to_Pharmacophore_graph(self, mol, cliques, edges, cliques_prop, normalize=True):
        clique_size = len(cliques)

        clique_index = list(range(clique_size))
        clique_attr = []
        for idx in range(len(cliques)):
            emb_0 = self.matrix[cliques[idx]].mean(0).tolist()
            emb_1 = self.getCliqueFeatures_pharmacophore(cliques[idx], edges, idx, cliques_prop[idx], mol)
            if normalize and sum(emb_1) != 0:
                clique_attr.append(emb_0 + list(emb_1 / sum(emb_1)))
            else:
                clique_attr.append(emb_0 + emb_1.tolist())

        cliqueedge_index = edges
        cliqueedge_attr = []
        for idx in range(len(edges)):
            cliqueedge_features = self.getCliqueEdgeFeatures_pharmacophore(edges[idx], cliques, idx, cliques_prop, mol)
            if normalize and sum(cliqueedge_features) != 0:
                cliqueedge_attr.append(list(cliqueedge_features) / sum(cliqueedge_features))
            else:
                cliqueedge_attr.append(cliqueedge_features)

        return clique_size, clique_index, clique_attr, cliqueedge_index, cliqueedge_attr

    """生成连接树图"""

    def getJunctionTreeGraph(self, mol):
        n_atoms = mol.GetNumAtoms()
        if n_atoms == 1:
            return [[0]], []

        cliques = []
        for bond in mol.GetBonds():
            a1 = bond.GetBeginAtom().GetIdx()
            a2 = bond.GetEndAtom().GetIdx()
            if not bond.IsInRing():
                cliques.append([a1, a2])

        ssr = [list(x) for x in Chem.GetSymmSSSR(mol)]
        cliques.extend(ssr)

        nei_list = [[] for i in range(n_atoms)]
        for i in range(len(cliques)):
            for atom in cliques[i]:
                nei_list[atom].append(i)

        # Merge rings that share more than 2 atoms as they form bridged compounds.
        for i in range(len(cliques)):
            if len(cliques[i]) <= 2: continue
            for atom in cliques[i]:
                for j in nei_list[atom]:
                    if i >= j or len(cliques[j]) <= 2: continue
                    inter = set(cliques[i]) & set(cliques[j])
                    if len(inter) > 2:
                        cliques[i].extend(cliques[j])
                        cliques[i] = list(set(cliques[i]))
                        cliques[j] = []

        cliques = [c for c in cliques if len(c) > 0]
        nei_list = [[] for i in range(n_atoms)]
        for i in range(len(cliques)):
            for atom in cliques[i]:
                nei_list[atom].append(i)

        # Build edges and add singleton cliques
        edges = defaultdict(int)
        for atom in range(n_atoms):
            if len(nei_list[atom]) <= 1:
                continue
            cnei = nei_list[atom]
            # Number of bond clusters that the atom lies in.
            bonds = [c for c in cnei if len(cliques[c]) == 2]
            # Number of ring clusters that the atom lies in.
            rings = [c for c in cnei if len(cliques[c]) > 4]
            # In general, if len(cnei) >= 3, a singleton should be added,
            # but 1 bond + 2 ring is currently not dealt with.
            if len(bonds) > 2 or (len(bonds) == 2 and len(cnei) > 2):
                cliques.append([atom])
                c2 = len(cliques) - 1
                for c1 in cnei:
                    edges[(c1, c2)] = 1
            # at least 1 bond connect to at least 2 rings
            elif len(rings) >= 2 and len(bonds) >= 1:
                cliques.append([atom])
                c2 = len(cliques) - 1
                for c1 in cnei:
                    edges[(c1, c2)] = MST_MAX_WEIGHT - 1
            # Multiple (n>2) complex rings
            elif len(rings) > 2:
                cliques.append([atom])
                c2 = len(cliques) - 1
                for c1 in cnei:
                    edges[(c1, c2)] = MST_MAX_WEIGHT - 1
            else:
                for i in range(len(cnei)):
                    for j in range(i + 1, len(cnei)):
                        c1, c2 = cnei[i], cnei[j]
                        inter = set(cliques[c1]) & set(cliques[c2])
                        if edges[(c1, c2)] < len(inter):
                            # cnei[i] < cnei[j] by construction ?
                            edges[(c1, c2)] = len(inter)
                            edges[(c2, c1)] = len(inter)

                            # check isolated single atom
        for atom in mol.GetAtoms():
            if atom.GetIdx() not in nei_list[atom.GetIdx()] and len(atom.GetBonds()) == 0:
                cliques.append([atom.GetIdx()])
                nei_list[atom.GetIdx()].append(len(cliques) - 1)

        edges = [u + (MST_MAX_WEIGHT - v,) for u, v in edges.items()]
        if len(edges) == 0:
            return cliques, edges

        # Compute Maximum Spanning Tree
        row, col, data = zip(*edges)
        n_clique = len(cliques)
        clique_graph = csr_matrix((data, (row, col)), shape=(n_clique, n_clique))
        junc_tree = minimum_spanning_tree(clique_graph)
        row, col = junc_tree.nonzero()
        # edges = [(row[i],col[i]) for i in range(len(row))]
        edges = []
        for i in range(len(row)):
            edges.append((row[i], col[i]))
            edges.append((col[i], row[i]))
        return cliques, edges

    def getCliqueFeatures_JunctionTree(self, clique, edges, clique_idx, mol, normalize=True):
        # number of node features (83)
        NumEachAtomDict = {a: 0 for a in definedAtom}
        NumEachBondDict = {b: 0 for b in definedBond}

        # number of atoms
        NumAtoms = len(clique)
        # number of edges
        NumEdges = 0
        for edge in edges:
            if clique_idx == edge[0] or clique_idx == edge[1]:
                NumEdges += 1
        # number of Hs
        NumHs = 0
        for atom in mol.GetAtoms():
            if atom.GetAtomMapNum() in clique:
                NumHs += atom.GetTotalNumHs()
                # number of each atom
                sb = atom.GetSymbol()
                if sb in NumEachAtomDict:
                    NumEachAtomDict[sb] += 1
                else:
                    NumEachAtomDict['Unknown'] += 1
        # is ring
        IsRing = 0
        if len(clique) > 2:
            IsRing = 1
        # is bond
        IsBond = 0
        if len(clique) == 2:
            IsBond = 1
        # number of each bond
        for bond in mol.GetBonds():
            if bond.GetBeginAtomIdx() in clique and bond.GetEndAtomIdx() in clique:
                bt = bond.GetBondType()
                NumEachBondDict[bt] += 1

        # convert number of each atom
        if sum(list(NumEachAtomDict.values())) != 0 and normalize:
            NumEachAtom = [float(i) / sum(list(NumEachAtomDict.values())) for i in list(NumEachAtomDict.values())]
        else:
            NumEachAtom = [int(i) for i in list(NumEachAtomDict.values())]
        # convert number of each bond
        if sum(list(NumEachBondDict.values())) != 0 and normalize:
            NumEachBond = [float(i) / sum(list(NumEachBondDict.values())) for i in list(NumEachBondDict.values())]
        else:
            NumEachBond = [int(i) for i in list(NumEachBondDict.values())]

        return np.array(
            one_of_k_encoding_unk(NumAtoms, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
            one_of_k_encoding_unk(NumEdges, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
            one_of_k_encoding_unk(NumHs, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
            [IsRing] +
            [IsBond] +
            NumEachAtom +
            NumEachBond
        )

    def getCliqueEdgeFeatures_JunctionTree(self, edge, clique, idx, mol):
        # number of edge features (6)
        len_begin = len(clique[edge[0]])
        len_end = len(clique[edge[1]])

        EdgeType = 5
        if len_begin == 1:
            begin_type = 'atom'
        elif len_begin > 2:
            begin_type = 'ring'
        else:
            begin_type = 'bond'
        if len_end == 1:
            end_type = 'atom'
        elif len_end > 2:
            end_type = 'ring'
        else:
            end_type = 'bond'

        definedEdgeType = {('atom', 'atom'): 0,
                           ('atom', 'bond'): 1,
                           ('atom', 'ring'): 2,
                           ('bond', 'atom'): 1,
                           ('bond', 'bond'): 3,
                           ('bond', 'ring'): 4,
                           ('ring', 'atom'): 2,
                           ('ring', 'bond'): 4,
                           ('ring', 'ring'): 5}
        EdgeType = definedEdgeType[(begin_type, end_type)]

        return np.array(one_of_k_encoding(EdgeType, list(set(definedEdgeType.values()))))

    def mol_to_JunctionTree_graph(self, mol, cliques, edges, normalize=True):
        clique_size = len(cliques)

        clique_index = list(range(clique_size))
        clique_attr = []
        for idx in range(len(cliques)):
            emb_0 = self.matrix[cliques[idx]].mean(0).tolist()
            emb_1 = self.getCliqueFeatures_JunctionTree(cliques[idx], edges, idx, mol, normalize)
            if normalize and sum(emb_1) != 0:
                clique_attr.append(emb_0 + list(emb_1 / sum(emb_1)))
            else:
                clique_attr.append(emb_0 + emb_1.tolist())

        cliqueedge_index = edges
        cliqueedge_attr = []
        for idx in range(len(edges)):
            cliqueedge_features = self.getCliqueEdgeFeatures_JunctionTree(edges[idx], cliques, idx, mol)
            if normalize and sum(cliqueedge_features) != 0:
                cliqueedge_attr.append(list(cliqueedge_features) / sum(cliqueedge_features))
            else:
                cliqueedge_attr.append(cliqueedge_features)

        return clique_size, clique_index, clique_attr, cliqueedge_index, cliqueedge_attr


class MolGraphSet_our_Multisub(Dataset):
    def __init__(self, df, target, filename, protrain_model="Mol2Vec", log=print, node_types="sfp"):
        # self.data = df.head(100)
        self.data = df
        self.filename = filename
        self.smiles = []
        self.mols = []
        self.labels = []
        self.graphs = []
        self.node_types = [i for i in node_types]
        self.edge_types = {
            "a": ("a", "b", "a"),  # 原子节点关系 bind
            "s": ("s", "c", "s"),  # Brics子结构关系 connect
            "f": ("f", "d", "f"),  # Function Group子结构关系 connect
            "p": ("p", "e", "p"),  # Pharmacophore子结构关系 connect
            # "m":("m","i","m"),  # 分子到分析 identity
            "as": ("a", "j_s", "s"),  # 原子到Brics子结构关系 junction
            "af": ("a", "j_f", "f"),  # 原子到Function Group子结构关系 junction
            "ap": ("a", "j_p", "p"),  # 原子到Pharmacophore子结构关系 junction
            "sm": ("s", "i_s", "m"),  # 子结构到分子关系 donate
            "fm": ("f", "i_f", "m"),  # 子结构到分子关系 donate
            "pm": ("p", "i_p", "m"),  # 子结构到分子关系 donate
        }
        self.subg_dim = 448  # 300+27+83+6+115-83 去除joint tree
        self.edge_dim = 57  # 14+34+6+3+20-6
        if protrain_model == "Mol2Vec": # 使用mol2vec模型给每个原子初始化为300维向量，对子结构即原子的相加求平均也是300维向量，分子对所有原子相加求平均得300维向量
            self.Mol2Vec = word2vec.Word2Vec.load('./model_300dim.pkl')
            try:
                self.keys = set(self.Mol2Vec.wv.key_to_index.keys())
            except:
                self.keys = set(self.Mol2Vec.wv.vocab.keys())
            self.unseen = 'UNK'
            self.unseen_vec = self.Mol2Vec.wv.word_vec(self.unseen)
        error_smiles = []
        for i, row in tqdm(self.data.iterrows(), total=self.data.shape[0]):
            smiles = row['smiles']
            label = row[target].values.astype(float)
            try:
                # if "." in smiles:
                #     mol = get_main_mol(smiles)
                # else:
                #     mol = Chem.MolFromSmiles(smiles)
                mol = Chem.MolFromSmiles(smiles)
                atom_num = mol.GetNumAtoms()
                if mol != None:
                    if atom_num >= 3:
                        self.matrix = Atom2Substructure(mol, 1, self.Mol2Vec, self.keys, self.unseen_vec)
                        try:
                            g = self.Mol2HeteroGraph(mol)
                            self.smiles.append(smiles)
                            self.mols.append(mol)
                            self.graphs.append(g)
                            self.labels.append(label)
                        except:
                            error_smiles.append(smiles)
                    else:
                        continue
                        # log(f"{smiles} too small")
                else:
                    error_smiles.append(smiles)
            except:
                error_smiles.append(smiles)
        print(len(error_smiles) / self.data.shape[0])
        # except Exception as e:
        #     log(e, 'invalid', smiles)

    def __len__(self):
        return len(self.mols)

    def __getitem__(self, idx):

        return self.smiles[idx], self.graphs[idx], self.labels[idx]

    def Mol2HeteroGraph(self, mol):

        edges = {edge_type: [] for node_type, edge_type in self.edge_types.items()}
        node_features = {node_type: [] for node_type in ["a", "s", "f", "p", "m"]}
        edge_features = {edge_type: [] for node_type, edge_type in self.edge_types.items()}
        # """构建分子图"""
        # edges[self.edge_types["m"]].append([0, 0])
        """构建原子图"""
        for bond in mol.GetBonds():
            edges[self.edge_types["a"]].append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
            edges[self.edge_types["a"]].append([bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()])
        """生成原子节点和边特征"""
        f_atom = []
        for atom in mol.GetAtoms():
            # f_atom.append(atom_features(atom))
            f_atom.append(np.concatenate([self.matrix[atom.GetIdx()], np.array(atom_features(atom))]))
        node_features["a"] = f_atom
        a_bond_f = []
        for src, dst in edges[self.edge_types["a"]]:
            a_bond_f.append(bond_features(mol.GetBondBetweenAtoms(src, dst)))
        edge_features[self.edge_types["a"]] = a_bond_f  # dim=14

        for node_type in self.node_types:
            if node_type == 's':
                """构建子结构图"""
                result_ap, result_p = self.GetFragmentFeats(mol)
                reac_idx, bbr = GetBricsBonds(mol)
                for r in reac_idx:
                    begin = r[1]
                    end = r[2]
                    edges[self.edge_types["s"]].append([result_ap[begin], result_ap[end]])
                    edges[self.edge_types["s"]].append([result_ap[end], result_ap[begin]])
                for k, v in result_ap.items():
                    edges[self.edge_types["as"]].append([k, v])
                for v in set(result_ap.values()):
                    edges[self.edge_types["sm"]].append([v, 0])

                """生成子结构节点和边特征"""
                f_substruct = []
                for k, v in result_p.items():
                    f_substruct.append(v)
                n_feature = np.zeros((len(f_substruct), 327))  # 300(mol2vec生成的feature)+27(mmgx中brics生成的特征)
                n_feature[:, :] = f_substruct
                node_features["s"].extend(n_feature.tolist())
                s_bond_f = []
                if len(edges[self.edge_types["s"]]) > 1:
                    for src, dst in edges[self.edge_types["s"]]:
                        p0_g = src
                        p1_g = dst
                        for i in bbr:
                            p0 = result_ap[i[0][0]]
                            p1 = result_ap[i[0][1]]
                            if p0_g == p0 and p1_g == p1:
                                s_bond_f.append(i[1])
                    # edge_feature = edge_f_init(s_bond_f,node_type)
                    edge_features[self.edge_types["s"]].extend(np.array(s_bond_f))  # dim=34

            elif node_type == 'f':
                """构建官能团图"""
                cliques, f_edges, cliques_func, cliques_ring = self.getFunctionalGraph(mol)
                f_set = set()
                for r in f_edges:
                    edges[self.edge_types["f"]].append([r[0], r[1]])
                    f_set.add(r[0])
                    f_set.add(r[1])
                for f_index, b_list in enumerate(cliques):
                    for b in b_list:
                        edges[self.edge_types["af"]].append([b, f_index])
                for f in f_set:
                    edges[self.edge_types["fm"]].append([f, 0])

                """生成官能团节点和边特征"""
                clique_size, clique_index, clique_attr, cliqueedge_index, cliqueedge_attr = self.mol_to_Functional_graph(
                    mol,
                    cliques,
                    f_edges,
                    cliques_func,
                    cliques_ring,
                    True)
                n_feature = np.zeros((len(clique_attr), 300 + 115))
                n_feature[:, :] = clique_attr  # 300+115
                # n_feature[:, :300] = np.array(clique_attr)[:, :300]  # 300+27+83+6+115
                # n_feature[:, 416:] = np.array(clique_attr)[:, 300:]  # 300+27+83+6+115
                node_features["f"].extend(n_feature.tolist())
                # edge_feature = edge_f_init(cliqueedge_attr, node_type)
                edge_features[self.edge_types["f"]].extend(np.array(cliqueedge_attr))  # dim = 20

            elif node_type == 'p':
                """构建药效团图"""
                cliques, p_edges, cliques_prop = self.getPharmacophoreGraph(mol)
                p_set = set()
                for r in p_edges:
                    edges[self.edge_types["p"]].append([r[0], r[1]])
                    p_set.add(r[0])
                    p_set.add(r[1])
                atoms_num = mol.GetNumAtoms()
                cliques_ = [item for sublist in cliques for item in sublist]
                if max(cliques_) != (atoms_num - 1) or min(cliques_) != 0:
                    print(atoms_num, cliques)

                for p_index, b_list in enumerate(cliques):
                    for b in b_list:
                        edges[self.edge_types["ap"]].append([b, p_index])
                for p in p_set:
                    edges[self.edge_types["pm"]].append([p, 0])

                clique_size, clique_index, clique_attr, cliqueedge_index, cliqueedge_attr = self.mol_to_Pharmacophore_graph(
                    mol, cliques, p_edges, cliques_prop)
                n_feature = np.zeros((len(clique_attr), 300 + 6))
                n_feature[:, :] = clique_attr
                # n_feature[:, :300] = np.array(clique_attr)[:, :300]  # 300+27+83+6+115
                # n_feature[:, 410:416] = np.array(clique_attr)[:, 300:]  # 300+27+83+6+115
                node_features["p"].extend(n_feature.tolist())
                # edge_feature = edge_f_init(cliqueedge_attr, node_type)
                edge_features[self.edge_types["p"]].extend(np.array(cliqueedge_attr))  # dim=3



        """构建异构图"""
        g = dgl.heterograph(edges)  # 构建异构图
        # g = dgl.add_self_loop(g, etype="b")
        # g = dgl.add_self_loop(g, etype="c")
        """生成分子节点特征"""
        # f_mol = GetMolecularFeats(mol)
        f_mol = self.matrix.mean(0)
        g.nodes['m'].data['f'] = torch.FloatTensor([f_mol])

        # self.node_types.append("a")
        node_dims = {}
        for node_type in ["a", "s", "f", "p"]:
            max_len = max(len(sublist) for sublist in node_features[node_type])
            # padded_data = [sublist + [0] * (max_len - len(sublist)) for sublist in node_features[node_type]]
            f_node = torch.tensor(node_features[node_type], dtype=torch.float32)
            g.nodes[node_type].data['f'] = f_node
            node_dims[node_type] = len(f_node[0])
        f_node = torch.FloatTensor(node_features["a"])
        g.nodes["a"].data['f'] = f_node
        node_dims["a"] = len(f_node[0])


        """
        "a":("a","b","a"),  # 原子节点关系 bind
            "s":("s","c","s"),  # 子结构关系 connect
            "m":("m","i","m"),  # 分子到分析 identity
            "as":("a","j","s"), # 原子到子结构关系 junction
            "sm":("s","d","m"), # 子结构到分子关系 donate"""

        return g

    def Mol2AtomGraph(self, mol):

        edges = {k: [] for k in self.edge_types}
        node_features = {}
        edge_features = {}
        """构建分子图"""
        edges[self.edge_type["m"]].append([0, 0])
        """构建原子图"""
        for bond in mol.GetBonds():
            edges[self.edge_type["a"]].append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
            edges[self.edge_type["a"]].append([bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()])
        """生成原子节点和边特征"""
        f_atom = []
        for atom in mol.GetAtoms():
            f_atom.append(atom_features(atom))
        node_features["a"] = f_atom
        a_bond_f = []
        for src, dst in edges[self.edge_type["a"]]:
            a_bond_f.append(bond_features(mol.GetBondBetweenAtoms(src, dst)))
        edge_features[self.edge_type["a"]] = a_bond_f

        for node_type in self.node_types:
            if node_type == 's':
                """构建子结构图"""
                result_ap, result_p = self.GetFragmentFeats(mol)
                reac_idx, bbr = GetBricsBonds(mol)
                for r in reac_idx:
                    begin = r[1]
                    end = r[2]
                    edges[self.edge_type["s"]].append([result_ap[begin], result_ap[end]])
                    edges[self.edge_type["s"]].append([result_ap[end], result_ap[begin]])
                for k, v in result_ap.items():
                    edges[self.edge_type["as"]].append([k, v])
                for v in set(result_ap.values()):
                    edges[self.edge_type["sm"]].append([v, 0])
                """生成子结构节点和边特征"""
                f_substruct = []
                for k, v in result_p.items():
                    f_substruct.append(v)
                node_features["s"] = f_substruct
                s_bond_f = []
                for src, dst in edges[self.edge_type["s"]]:
                    p0_g = src
                    p1_g = dst
                    for i in bbr:
                        p0 = result_ap[i[0][0]]
                        p1 = result_ap[i[0][1]]
                        if p0_g == p0 and p1_g == p1:
                            s_bond_f.append(i[1])
                edge_features[self.edge_type["s"]] = s_bond_f

            elif node_type == 'f':
                """构建官能团图"""
                cliques, f_edges, cliques_func, cliques_ring = self.getFunctionalGraph(mol)
                f_set = set()
                for r in f_edges:
                    edges[self.edge_type["f"]].append([r[0], r[1]])
                    edges[self.edge_type["f"]].append([r[1], r[0]])
                    f_set.add(r[0])
                    f_set.add(r[1])
                for f_index, b_list in enumerate(cliques):
                    for b in b_list:
                        edges[self.edge_type["af"]].append([b, f_index])
                for f in f_set:
                    edges[self.edge_type["fm"]].append([f, 0])
                """生成官能团节点和边特征"""
                clique_size, clique_index, clique_attr, cliqueedge_index, cliqueedge_attr = self.mol_to_Functional_graph(
                    mol,
                    cliques,
                    f_edges,
                    cliques_func,
                    cliques_ring,
                    True)
                node_features["f"] = clique_attr
                edge_features[self.edge_type["f"]] = cliqueedge_attr

            elif node_type == 'p':
                """构建药效团图"""
                cliques, p_edges, cliques_prop = self.getPharmacophoreGraph(mol)
                p_set = set()
                for r in p_edges:
                    edges[self.edge_type["p"]].append([r[0], r[1]])
                    edges[self.edge_type["p"]].append([r[1], r[0]])
                    p_set.add(r[0])
                    p_set.add(r[1])
                atoms_num = mol.GetNumAtoms()
                cliques_ = [item for sublist in cliques for item in sublist]
                if max(cliques_) != (atoms_num - 1) or min(cliques_) != 0:
                    print(atoms_num, cliques)

                for p_index, b_list in enumerate(cliques):
                    for b in b_list:
                        edges[self.edge_type["ap"]].append([b, p_index])
                for p in p_set:
                    edges[self.edge_type["pm"]].append([p, 0])
                clique_size, clique_index, clique_attr, cliqueedge_index, cliqueedge_attr = self.mol_to_Pharmacophore_graph(
                    mol, cliques, p_edges, cliques_prop)
                node_features["p"] = clique_attr
                edge_features[self.edge_type["p"]] = cliqueedge_attr

            elif node_type == 'j':
                """构建连接树图"""
                cliques, j_edges = self.getJunctionTreeGraph(mol)
                j_set = set()
                for r in j_edges:
                    edges[self.edge_type["j"]].append([r[0], r[1]])
                    edges[self.edge_type["j"]].append([r[1], r[0]])
                    j_set.add(r[0])
                    j_set.add(r[1])
                for j_index, b_list in enumerate(cliques):
                    for b in b_list:
                        edges[self.edge_type["aj"]].append([b, j_index])
                for j in j_set:
                    edges[self.edge_type["jm"]].append([j, 0])
                clique_size, clique_index, clique_attr, cliqueedge_index, cliqueedge_attr = self.mol_to_JunctionTree_graph(
                    mol,
                    cliques,
                    j_edges,
                    normalize=True)
                node_features["j"] = clique_attr
                edge_features[self.edge_type["j"]] = cliqueedge_attr

        """构建异构图"""
        g = dgl.heterograph(edges)  # 构建异构图
        """生成分子节点特征"""
        # f_mol = GetMolecularFeats(mol)
        f_mol = self.matrix.mean(0)
        g.nodes['m'].data['f'] = torch.FloatTensor([f_mol])

        # self.node_types.append("a")
        node_dims = {}
        for node_type in self.node_types:
            f_node = torch.tensor(node_features[node_type], dtype=torch.float32)
            g.nodes[node_type].data['f'] = f_node
            node_dims[node_type] = len(f_node[0])
        f_node = torch.FloatTensor(node_features["a"])
        g.nodes["a"].data['f'] = f_node
        node_dims["a"] = len(f_node[0])

        node_nums = {}
        for node_type in self.node_types:
            node_nums[node_type] = g.nodes[node_type].data['f'].size()[0]
        node_nums["a"] = g.nodes["a"].data['f'].size()[0]
        max_dim = max(node_dims.values())

        return g

    def GetFragmentFeats(self, mol):
        break_bonds = [mol.GetBondBetweenAtoms(i[0][0], i[0][1]).GetIdx() for i in FindBRICSBonds(mol)]
        if break_bonds == []:
            tmp = mol
        else:
            tmp = Chem.FragmentOnBonds(mol, break_bonds, addDummies=False)
        frags_idx_lst = Chem.GetMolFrags(tmp)
        result_ap = {}
        result_p = {}
        pharm_id = 0
        for frag_idx in frags_idx_lst:  # 片段
            for atom_id in frag_idx:
                result_ap[atom_id] = pharm_id
            try:
                mol_pharm = Chem.MolFromSmiles(Chem.MolFragmentToSmiles(mol, frag_idx))
                emb_0 = self.matrix[list(frag_idx)].mean(0).tolist()
                # emb_0 = maccskeys_emb(mol_pharm)  # MaccsKey指纹，二进制长度
                emb_1 = pharm_property_types_feats(mol_pharm)  # 27
            except Exception:
                emb_0 = [0 for i in range(300)]
                emb_1 = [0 for i in range(27)]
            result_p[pharm_id] = emb_0 + emb_1

            pharm_id += 1
        return result_ap, result_p

    """生成官能团图"""

    def getFunctionalGraph(self, mol):
        n_atoms = mol.GetNumAtoms()

        # functional group
        funcGroupDict = dict()
        for i in range(fparams.GetNumFuncGroups()):
            funcGroupDict[i] = list(mol.GetSubstructMatches(fparams.GetFuncGroup(i)))

        # edit #27 <-> #29
        temp = funcGroupDict[27]
        funcGroupDict[27] = funcGroupDict[29]
        funcGroupDict[29] = temp

        cliques = []
        cliques_ring = {}  # node group in ring
        cliques_func = {}  # node group in func
        seen_func = {}  # node seen in func
        group_num = 0

        # extract functional from substructure match
        for f in funcGroupDict:
            for l in funcGroupDict[f]:
                if not (all(ll in seen_func for ll in l)):
                    cliques.append(list(l))
                    for ll in l:
                        if ll in seen_func or ll in cliques_func:
                            cliques_func[ll].append(f)
                            seen_func[ll].append(group_num)
                        else:
                            cliques_func[ll] = [f]
                            seen_func[ll] = [group_num]
                group_num += 1

        # extract bond which not in functional
        for bond in mol.GetBonds():
            a1 = bond.GetBeginAtom().GetIdx()
            a2 = bond.GetEndAtom().GetIdx()
            bt = bond.GetBondType()
            if not bond.IsInRing():
                if (a1 not in seen_func) or (a2 not in seen_func):
                    cliques.append([a1, a2])
                elif a1 in seen_func and a2 in seen_func and len(set(seen_func[a1]) & set(seen_func[a2])) == 0:
                    cliques.append([a1, a2])

        # extract ring
        ssr = [list(x) for x in Chem.GetSymmSSSR(mol)]
        cliques.extend(ssr)
        cliques_ring = ssr

        nei_list = [[] for i in range(n_atoms)]
        for i in range(len(cliques)):
            for atom in cliques[i]:
                nei_list[atom].append(i)



        cliques = [c for c in cliques if len(c) > 0]
        nei_list = [[] for i in range(n_atoms)]
        for i in range(len(cliques)):
            for atom in cliques[i]:
                nei_list[atom].append(i)

        # Build edges and add singleton cliques
        edges = defaultdict(int)
        for atom in range(n_atoms):
            if len(nei_list[atom]) <= 1:
                continue
            cnei = nei_list[atom]
            # Number of bond clusters that the atom lies in.
            bonds = [c for c in cnei if len(cliques[c]) == 2]
            # Number of ring clusters that the atom lies in.
            funcring = [c for c in cnei if len(cliques[c]) > 2]



            for i in range(len(cnei)):
                for j in range(i + 1, len(cnei)):
                    c1, c2 = cnei[i], cnei[j]
                    inter = set(cliques[c1]) & set(cliques[c2])
                    if edges[(c1, c2)] < len(inter):
                        # cnei[i] < cnei[j] by construction ?
                        edges[(c1, c2)] = len(inter)
                        edges[(c2, c1)] = len(inter)

        # check isolated single atom
        for atom in mol.GetAtoms():
            if atom.GetIdx() not in nei_list[atom.GetIdx()] and len(atom.GetBonds()) == 0:
                cliques.append([atom.GetIdx()])
                nei_list[atom.GetIdx()].append(len(cliques) - 1)

        edges = [i for i in edges]

        return cliques, edges, cliques_func, cliques_ring

    def getCliqueFeatures_funcGroup(self, clique, edges, clique_idx, cliques_func, cliques_ring, mol):
        # number of node features (115)
        funcType = [0 for f in range(len(range(fparams.GetNumFuncGroups())))]  # no unknown
        funcRingTypeList = range(len(definedRing))  # (only aromatic)
        funcRingTypeOtherList = range(len(definedRing))  # (other bonds)
        funcRingTypeSizeList = [3, 4, 5, 6, 7, 8, 9, 10]  # unknown ring size 3-9 and >9
        funcBondTypeList = range(len(definedFuncBond))  # included unknown
        # atomTypeList = range(len(definedAtom)) # included unknown

        ringType = one_of_k_encoding_none(None, funcRingTypeList)
        ringTypeOther = one_of_k_encoding_none(None, funcRingTypeOtherList)
        ringTypeSize = one_of_k_encoding_none(None, funcRingTypeSizeList)
        funcBondType = one_of_k_encoding_none(None, funcBondTypeList)
        # atomType = atomType = one_of_k_encoding_none(None, atomTypeList)

        # functional group
        func_found = False
        if all(c in cliques_func for c in clique):
            funcGroup = [cliques_func[c] for c in clique]
            intersect = funcGroup[0]
            for f in funcGroup:
                intersect = set(set(intersect) & set(f))
            for i in list(intersect):
                funcType[i] = 1
                func_found = True
            if func_found:  # func, not ring not bond
                ringType = one_of_k_encoding_none(None, funcRingTypeList)
                ringTypeOther = one_of_k_encoding_none(None, funcRingTypeOtherList)
                ringTypeSize = one_of_k_encoding_none(None, funcRingTypeSizeList)
                funcBondType = one_of_k_encoding_none(None, funcBondTypeList)
        # ring type
        if len(clique) > 2 and not func_found:
            if clique in cliques_ring:
                new_mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))  # not kekulize
                smarts = Chem.MolFragmentToSmarts(new_mol, clique)
                ring_found = False
                for ring in definedRing:
                    mol_ring = Chem.MolFromSmarts(ring)
                    mol_smart = Chem.MolFromSmarts(smarts)
                    t1 = topology_checker(mol_ring)
                    t2 = topology_checker(mol_smart)
                    if len(mol_smart.GetSubstructMatches(mol_ring)) != 0:
                        ringType = one_of_k_encoding_unk(definedRing.index(ring), funcRingTypeList)
                        ringTypeOther = one_of_k_encoding_none(None, funcRingTypeOtherList)
                        ringTypeSize = one_of_k_encoding_none(None, funcRingTypeSizeList)
                        funcBondType = one_of_k_encoding_none(None, funcBondTypeList)
                        ring_found = True
                        break
                    elif is_isomorphic(t1, t2):
                        ringType = one_of_k_encoding_unk(definedRing.index(ring), funcRingTypeList)
                        ringTypeOther = one_of_k_encoding_none(None, funcRingTypeOtherList)
                        ringTypeSize = one_of_k_encoding_none(None, funcRingTypeSizeList)
                        funcBondType = one_of_k_encoding_none(None, funcBondTypeList)
                        ring_found = True
                        break
                    elif is_isomorphic_atom(t1, t2):
                        ringType = one_of_k_encoding_none(None, funcRingTypeList)
                        ringTypeOther = one_of_k_encoding_unk(definedRing.index(ring), funcRingTypeOtherList)
                        ringTypeSize = one_of_k_encoding_none(None, funcRingTypeSizeList)
                        funcBondType = one_of_k_encoding_none(None, funcBondTypeList)
                        ring_found = True
                        break
                if not ring_found:  # unknown ring
                    mol_smart = Chem.MolFromSmarts(smarts)
                    ringType = one_of_k_encoding_none(None, funcRingTypeList)
                    ringTypeOther = one_of_k_encoding_none(None, funcRingTypeOtherList)
                    ringTypeSize = one_of_k_encoding_unk(mol_smart.GetNumAtoms(), funcRingTypeSizeList)
                    funcBondType = one_of_k_encoding_none(None, funcBondTypeList)
            else:  # not ring not bond
                ringType = one_of_k_encoding_none(None, funcRingTypeList)
                ringTypeOther = one_of_k_encoding_none(None, funcRingTypeOtherList)
                ringTypeSize = one_of_k_encoding_none(None, funcRingTypeSizeList)
                funcBondType = one_of_k_encoding_none(None, funcBondTypeList)
        # bond type
        if len(clique) == 2 and not func_found:
            bond_found = False
            for bond in mol.GetBonds():
                b = bond.GetBondType()
                atom1 = bond.GetBeginAtom()
                atom2 = bond.GetEndAtom()
                a1 = atom1.GetIdx()
                a2 = atom2.GetIdx()
                a1_s = atom1.GetSymbol()
                a2_s = atom2.GetSymbol()
                if [a1, a2] == clique or [a2, a1] == clique:
                    if (a1_s, b, a2_s) in definedFuncBond:
                        funcBondType = one_of_k_encoding_unk(definedFuncBond.index((a1_s, b, a2_s)), funcBondTypeList)
                        ringType = one_of_k_encoding_none(None, funcRingTypeList)
                        ringTypeOther = one_of_k_encoding_none(None, funcRingTypeOtherList)
                        ringTypeSize = one_of_k_encoding_none(None, funcRingTypeSizeList)
                        bond_found = True
                        break
                    elif (a2_s, b, a1_s) in definedFuncBond:
                        funcBondType = one_of_k_encoding_unk(definedFuncBond.index((a2_s, b, a1_s)), funcBondTypeList)
                        ringType = one_of_k_encoding_none(None, funcRingTypeList)
                        ringTypeOther = one_of_k_encoding_none(None, funcRingTypeOtherList)
                        ringTypeSize = one_of_k_encoding_none(None, funcRingTypeSizeList)
                        bond_found = True
                        break
            if not bond_found:  # unknown bond
                ringType = one_of_k_encoding_none(None, funcRingTypeList)
                ringTypeOther = one_of_k_encoding_none(None, funcRingTypeOtherList)
                ringTypeSize = one_of_k_encoding_none(None, funcRingTypeSizeList)
                funcBondType = one_of_k_encoding_unk(None, funcBondTypeList)


        return np.array(funcType + ringType + ringTypeOther + ringTypeSize + funcBondType)

    def getCliqueEdgeFeatures_funcGroup(self, edge, clique, edge_idx, cliques_func, cliques_ring, mol):
        # number of edge features (10)+10 = (20) # (10)+12 = (22)
        begin = clique[edge[0]]
        end = clique[edge[1]]

        if all(c in cliques_func for c in begin):
            begin_type = 'func'
        elif len(begin) > 2:
            begin_type = 'ring'
        elif len(begin) == 1:
            begin_type = 'atom'
        else:
            begin_type = 'bond'

        if all(c in cliques_func for c in end):
            end_type = 'func'
        elif len(end) > 2:
            end_type = 'ring'
        elif len(end) == 1:
            end_type = 'atom'
        else:
            end_type = 'bond'

        intersect = len(set(begin) & set(end))

        begin_atom = 0
        end_atom = 0
        if intersect == 1:
            a1 = list(set(begin) & set(end))[0]
            a2 = list(set(begin) & set(end))[0]
            begin_atom = sorted(begin).index(a1) + 1
            end_atom = sorted(end).index(a2) + 1
        # in case, more than 2 common atoms
        else:
            begin_atom = 0
            end_atom = 0

        definedEdgeType = {('atom', 'atom'): 0,
                           ('atom', 'bond'): 1,
                           ('atom', 'ring'): 2,
                           ('atom', 'func'): 3,
                           ('bond', 'atom'): 1,
                           ('bond', 'bond'): 4,
                           ('bond', 'ring'): 5,
                           ('bond', 'func'): 6,
                           ('ring', 'atom'): 2,
                           ('ring', 'bond'): 5,
                           ('ring', 'ring'): 7,
                           ('ring', 'func'): 8,
                           ('func', 'atom'): 3,
                           ('func', 'bond'): 6,
                           ('func', 'ring'): 8,
                           ('func', 'func'): 9}
        edgeType = definedEdgeType[(begin_type, end_type)]


        return np.array(one_of_k_encoding(edgeType, list(set(definedEdgeType.values()))) +
                        one_of_k_encoding_unk(intersect, list(range(10))))

    def mol_to_Functional_graph(self, mol, cliques, edges, cliques_func, cliques_ring, normalize=True):
        clique_size = len(cliques)

        clique_index = list(range(clique_size))
        clique_attr = []
        for idx in range(len(cliques)):
            emb_0 = self.matrix[cliques[idx]].mean(0).tolist()
            emb_1 = self.getCliqueFeatures_funcGroup(cliques[idx], edges, idx, cliques_func, cliques_ring,
                                                     mol)
            if normalize and sum(emb_1) != 0:
                clique_attr.append(emb_0 + list(emb_1 / sum(emb_1)))
            else:
                clique_attr.append(emb_0 + emb_1.tolist())

        cliqueedge_index = edges
        cliqueedge_attr = []
        for idx in range(len(edges)):
            cliqueedge_features = self.getCliqueEdgeFeatures_funcGroup(edges[idx], cliques, idx, cliques_func,
                                                                       cliques_ring, mol)
            if normalize and sum(cliqueedge_features) != 0:
                cliqueedge_attr.append(list(cliqueedge_features) / sum(cliqueedge_features))
            else:
                cliqueedge_attr.append(cliqueedge_features)

        return clique_size, clique_index, clique_attr, cliqueedge_index, cliqueedge_attr

    """生成药效团图"""

    def getPharmacophoreGraph(self, mol):
        mol = mol_with_atom_index(mol)
        mol_g = Chem.rdReducedGraphs.GenerateMolExtendedReducedGraph(mol)
        mol_g.UpdatePropertyCache(False)
        mapping_atom = {a.GetAtomMapNum(): i for i, a in enumerate(mol_g.GetAtoms())}

        cliques = []
        cliques_prop = []

        ring_8 = [list(x) for x in Chem.GetSymmSSSR(mol) if len(list(x)) < 8]
        ring_B8 = [list(x) for x in Chem.GetSymmSSSR(mol) if len(list(x)) >= 8]

        # add more 8-atom ring
        if len(ring_B8) > 0:
            rwmol_g = Chem.RWMol(mol_g)
            for rb8 in ring_B8:
                new_a = rwmol_g.AddAtom(Chem.Atom(0))
                rwmol_g.GetAtomWithIdx(new_a).SetProp('_ErGAtomTypes', '')
                for rb8_a in rb8:
                    if rb8_a in mapping_atom:
                        rwmol_g.AddBond(new_a, mapping_atom[rb8_a], Chem.BondType.SINGLE)
            mol_g = rwmol_g

        # display(mol_g)
        num_ring_8 = 0
        num_ring_B8 = 0

        for atom in mol_g.GetAtoms():
            if atom.GetAtomMapNum() == 0 and atom.GetSymbol() == '*' and 5 in list(
                    atom.GetPropsAsDict()['_ErGAtomTypes']):
                cliques.append(ring_8[num_ring_8])
                cliques_prop.append(list(atom.GetPropsAsDict()['_ErGAtomTypes']))
                num_ring_8 += 1
            elif atom.GetAtomMapNum() == 0 and atom.GetSymbol() == '*' and 4 in list(
                    atom.GetPropsAsDict()['_ErGAtomTypes']):
                cliques.append(ring_8[num_ring_8])
                cliques_prop.append(list(atom.GetPropsAsDict()['_ErGAtomTypes']))
                num_ring_8 += 1
            elif atom.GetAtomMapNum() == 0 and atom.GetSymbol() == '*':
                cliques.append(ring_B8[num_ring_B8])
                cliques_prop.append(list(atom.GetPropsAsDict()['_ErGAtomTypes']))
                num_ring_B8 += 1
            else:
                cliques.append([atom.GetAtomMapNum() - 1])
                cliques_prop.append(list(atom.GetPropsAsDict()['_ErGAtomTypes']))

        edges = []
        for bond in mol_g.GetBonds():
            a1 = bond.GetBeginAtomIdx()
            a2 = bond.GetEndAtomIdx()
            edges.append((a1, a2))
            edges.append((a2, a1))

        cliques_ = []
        for clique in cliques:
            cliques_.append([atom_index for atom_index in clique])

        return cliques_, edges, cliques_prop

    def getCliqueFeatures_pharmacophore(self, clique, edges, clique_idx, cliques_prop, mol):
        # number of node features (6)
        pharmacophore = np.zeros(6)
        for p in cliques_prop:
            pharmacophore[p] = 1

        return np.array(pharmacophore)

    def getCliqueEdgeFeatures_pharmacophore(self, edge, clique, edge_idx, cliques_prop, mol):
        # number of edge features (3)
        begin = cliques_prop[edge[0]]
        end = cliques_prop[edge[1]]

        if len(begin) == 0:
            begin_type = 'none'
        else:
            begin_type = 'phar'

        if len(end) == 0:
            end_type = 'none'
        else:
            end_type = 'phar'

        definedEdgeType = {('none', 'none'): 0,
                           ('none', 'phar'): 1,
                           ('phar', 'none'): 1,
                           ('phar', 'phar'): 2, }
        EdgeType = definedEdgeType[(begin_type, end_type)]

        return np.array(one_of_k_encoding(EdgeType, list(set(definedEdgeType.values()))))

    def mol_to_Pharmacophore_graph(self, mol, cliques, edges, cliques_prop, normalize=True):
        clique_size = len(cliques)

        clique_index = list(range(clique_size))
        clique_attr = []
        for idx in range(len(cliques)):
            emb_0 = self.matrix[cliques[idx]].mean(0).tolist()
            emb_1 = self.getCliqueFeatures_pharmacophore(cliques[idx], edges, idx, cliques_prop[idx], mol)
            if normalize and sum(emb_1) != 0:
                clique_attr.append(emb_0 + list(emb_1 / sum(emb_1)))
            else:
                clique_attr.append(emb_0 + emb_1.tolist())

        cliqueedge_index = edges
        cliqueedge_attr = []
        for idx in range(len(edges)):
            cliqueedge_features = self.getCliqueEdgeFeatures_pharmacophore(edges[idx], cliques, idx, cliques_prop, mol)
            if normalize and sum(cliqueedge_features) != 0:
                cliqueedge_attr.append(list(cliqueedge_features) / sum(cliqueedge_features))
            else:
                cliqueedge_attr.append(cliqueedge_features)

        return clique_size, clique_index, clique_attr, cliqueedge_index, cliqueedge_attr

    """生成连接树图"""

    def getJunctionTreeGraph(self, mol):
        n_atoms = mol.GetNumAtoms()
        if n_atoms == 1:
            return [[0]], []

        cliques = []
        for bond in mol.GetBonds():
            a1 = bond.GetBeginAtom().GetIdx()
            a2 = bond.GetEndAtom().GetIdx()
            if not bond.IsInRing():
                cliques.append([a1, a2])

        ssr = [list(x) for x in Chem.GetSymmSSSR(mol)]
        cliques.extend(ssr)

        nei_list = [[] for i in range(n_atoms)]
        for i in range(len(cliques)):
            for atom in cliques[i]:
                nei_list[atom].append(i)

        # Merge rings that share more than 2 atoms as they form bridged compounds.
        for i in range(len(cliques)):
            if len(cliques[i]) <= 2: continue
            for atom in cliques[i]:
                for j in nei_list[atom]:
                    if i >= j or len(cliques[j]) <= 2: continue
                    inter = set(cliques[i]) & set(cliques[j])
                    if len(inter) > 2:
                        cliques[i].extend(cliques[j])
                        cliques[i] = list(set(cliques[i]))
                        cliques[j] = []

        cliques = [c for c in cliques if len(c) > 0]
        nei_list = [[] for i in range(n_atoms)]
        for i in range(len(cliques)):
            for atom in cliques[i]:
                nei_list[atom].append(i)

        # Build edges and add singleton cliques
        edges = defaultdict(int)
        for atom in range(n_atoms):
            if len(nei_list[atom]) <= 1:
                continue
            cnei = nei_list[atom]
            # Number of bond clusters that the atom lies in.
            bonds = [c for c in cnei if len(cliques[c]) == 2]
            # Number of ring clusters that the atom lies in.
            rings = [c for c in cnei if len(cliques[c]) > 4]
            # In general, if len(cnei) >= 3, a singleton should be added,
            # but 1 bond + 2 ring is currently not dealt with.
            if len(bonds) > 2 or (len(bonds) == 2 and len(cnei) > 2):
                cliques.append([atom])
                c2 = len(cliques) - 1
                for c1 in cnei:
                    edges[(c1, c2)] = 1
            # at least 1 bond connect to at least 2 rings
            elif len(rings) >= 2 and len(bonds) >= 1:
                cliques.append([atom])
                c2 = len(cliques) - 1
                for c1 in cnei:
                    edges[(c1, c2)] = MST_MAX_WEIGHT - 1
            # Multiple (n>2) complex rings
            elif len(rings) > 2:
                cliques.append([atom])
                c2 = len(cliques) - 1
                for c1 in cnei:
                    edges[(c1, c2)] = MST_MAX_WEIGHT - 1
            else:
                for i in range(len(cnei)):
                    for j in range(i + 1, len(cnei)):
                        c1, c2 = cnei[i], cnei[j]
                        inter = set(cliques[c1]) & set(cliques[c2])
                        if edges[(c1, c2)] < len(inter):
                            # cnei[i] < cnei[j] by construction ?
                            edges[(c1, c2)] = len(inter)
                            edges[(c2, c1)] = len(inter)

                            # check isolated single atom
        for atom in mol.GetAtoms():
            if atom.GetIdx() not in nei_list[atom.GetIdx()] and len(atom.GetBonds()) == 0:
                cliques.append([atom.GetIdx()])
                nei_list[atom.GetIdx()].append(len(cliques) - 1)

        edges = [u + (MST_MAX_WEIGHT - v,) for u, v in edges.items()]
        if len(edges) == 0:
            return cliques, edges

        # Compute Maximum Spanning Tree
        row, col, data = zip(*edges)
        n_clique = len(cliques)
        clique_graph = csr_matrix((data, (row, col)), shape=(n_clique, n_clique))
        junc_tree = minimum_spanning_tree(clique_graph)
        row, col = junc_tree.nonzero()
        # edges = [(row[i],col[i]) for i in range(len(row))]
        edges = []
        for i in range(len(row)):
            edges.append((row[i], col[i]))
            edges.append((col[i], row[i]))
        return cliques, edges

    def getCliqueFeatures_JunctionTree(self, clique, edges, clique_idx, mol, normalize=True):
        # number of node features (83)
        NumEachAtomDict = {a: 0 for a in definedAtom}
        NumEachBondDict = {b: 0 for b in definedBond}

        # number of atoms
        NumAtoms = len(clique)
        # number of edges
        NumEdges = 0
        for edge in edges:
            if clique_idx == edge[0] or clique_idx == edge[1]:
                NumEdges += 1
        # number of Hs
        NumHs = 0
        for atom in mol.GetAtoms():
            if atom.GetAtomMapNum() in clique:
                NumHs += atom.GetTotalNumHs()
                # number of each atom
                sb = atom.GetSymbol()
                if sb in NumEachAtomDict:
                    NumEachAtomDict[sb] += 1
                else:
                    NumEachAtomDict['Unknown'] += 1
        # is ring
        IsRing = 0
        if len(clique) > 2:
            IsRing = 1
        # is bond
        IsBond = 0
        if len(clique) == 2:
            IsBond = 1
        # number of each bond
        for bond in mol.GetBonds():
            if bond.GetBeginAtomIdx() in clique and bond.GetEndAtomIdx() in clique:
                bt = bond.GetBondType()
                NumEachBondDict[bt] += 1

        # convert number of each atom
        if sum(list(NumEachAtomDict.values())) != 0 and normalize:
            NumEachAtom = [float(i) / sum(list(NumEachAtomDict.values())) for i in list(NumEachAtomDict.values())]
        else:
            NumEachAtom = [int(i) for i in list(NumEachAtomDict.values())]
        # convert number of each bond
        if sum(list(NumEachBondDict.values())) != 0 and normalize:
            NumEachBond = [float(i) / sum(list(NumEachBondDict.values())) for i in list(NumEachBondDict.values())]
        else:
            NumEachBond = [int(i) for i in list(NumEachBondDict.values())]

        return np.array(
            one_of_k_encoding_unk(NumAtoms, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
            one_of_k_encoding_unk(NumEdges, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
            one_of_k_encoding_unk(NumHs, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
            [IsRing] +
            [IsBond] +
            NumEachAtom +
            NumEachBond
        )

    def getCliqueEdgeFeatures_JunctionTree(self, edge, clique, idx, mol):
        # number of edge features (6)
        len_begin = len(clique[edge[0]])
        len_end = len(clique[edge[1]])

        EdgeType = 5
        if len_begin == 1:
            begin_type = 'atom'
        elif len_begin > 2:
            begin_type = 'ring'
        else:
            begin_type = 'bond'
        if len_end == 1:
            end_type = 'atom'
        elif len_end > 2:
            end_type = 'ring'
        else:
            end_type = 'bond'

        definedEdgeType = {('atom', 'atom'): 0,
                           ('atom', 'bond'): 1,
                           ('atom', 'ring'): 2,
                           ('bond', 'atom'): 1,
                           ('bond', 'bond'): 3,
                           ('bond', 'ring'): 4,
                           ('ring', 'atom'): 2,
                           ('ring', 'bond'): 4,
                           ('ring', 'ring'): 5}
        EdgeType = definedEdgeType[(begin_type, end_type)]

        return np.array(one_of_k_encoding(EdgeType, list(set(definedEdgeType.values()))))

    def mol_to_JunctionTree_graph(self, mol, cliques, edges, normalize=True):
        clique_size = len(cliques)

        clique_index = list(range(clique_size))
        clique_attr = []
        for idx in range(len(cliques)):
            emb_0 = self.matrix[cliques[idx]].mean(0).tolist()
            emb_1 = self.getCliqueFeatures_JunctionTree(cliques[idx], edges, idx, mol, normalize)
            if normalize and sum(emb_1) != 0:
                clique_attr.append(emb_0 + list(emb_1 / sum(emb_1)))
            else:
                clique_attr.append(emb_0 + emb_1.tolist())

        cliqueedge_index = edges
        cliqueedge_attr = []
        for idx in range(len(edges)):
            cliqueedge_features = self.getCliqueEdgeFeatures_JunctionTree(edges[idx], cliques, idx, mol)
            if normalize and sum(cliqueedge_features) != 0:
                cliqueedge_attr.append(list(cliqueedge_features) / sum(cliqueedge_features))
            else:
                cliqueedge_attr.append(cliqueedge_features)

        return clique_size, clique_index, clique_attr, cliqueedge_index, cliqueedge_attr


def create_dataloader_our(config, filename, shuffle=True, train=True, node_types="sfp"):
    data_df = pd.read_csv(os.path.join(config['data_path'], filename))
    data_lables = data_df.columns[1:]
    dataset = MolGraphSet_our_Multisub(data_df, data_lables, filename, node_types=node_types)
    if train:
        batch_size = config['batch_size']
        dataloader = GraphDataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True, num_workers=4)
    else:
        batch_size = config['batch_size']
        dataloader = GraphDataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False, num_workers=4)

    return dataloader


def random_split(load_path, save_dir, num_fold=5, sizes=[0.8, 0.1, 0.1], seed=0):
    df = pd.read_csv(load_path)
    n = len(df)
    os.makedirs(save_dir, exist_ok=True)
    torch.manual_seed(seed)
    for fold in range(num_fold):
        df = df.loc[torch.randperm(n)].reset_index(drop=True)
        train_size = int(sizes[0] * n)
        train_val_size = int((sizes[0] + sizes[1]) * n)
        train = df[:train_size]
        val = df[train_size:train_val_size]
        test = df[train_val_size:]
        train.to_csv(os.path.join(save_dir) + f'{seed}_fold_{fold}_train.csv', index=False)
        val.to_csv(os.path.join(save_dir) + f'{seed}_fold_{fold}_valid.csv', index=False)
        test.to_csv(os.path.join(save_dir) + f'{seed}_fold_{fold}_test.csv', index=False)


if __name__ == '__main__':
    for dataset_name in ["bbbp", "bace", "freesolv", "lipophilicity", "esol"]:
        dataset = MolGraphSet_our(pd.read_csv(f"./../../../Datasets/{dataset_name}.csv"), ["label"], node_types="sjpf")
        batch_size = 64

        dataloader = GraphDataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=4)

