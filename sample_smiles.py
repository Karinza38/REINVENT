import numpy as np
import torch
from model import RNN
from data_structs import Vocabulary, Experience
from utils import seq_to_smiles 
from rdkit.Chem import MolFromSmiles
from rdkit import rdBase
rdBase.DisableLog('rdApp.error')

def generate_smiles(n_smiles=500, restore_from="data/Prior.ckpt", voc_file="data/Voc", batch_size=128):
    """ 
    This function takes a checkpoint for a trained RNN and the vocabulary file and generates n_smiles new smiles strings.
    """
    n_smiles = n_smiles - n_smiles%64
    print("Generating %i smiles" % n_smiles)

    voc = Vocabulary(init_from_file=voc_file)
    generator = RNN(voc, batch_size)

    if torch.cuda.is_available():
        generator.rnn.load_state_dict(torch.load(restore_from))
    else:
        generator.rnn.load_state_dict(torch.load(restore_from, map_location=lambda storage, loc: storage))
    
    all_smiles = []
    for i in range(int(n_smiles/64)):
        sequences, _, _ = generator.sample(64)
        smiles = seq_to_smiles(sequences, voc)
        all_smiles += smiles

    return all_smiles

def check_unique_valid(smiles):

    """
    Gives the percentage of unique smiles string and what percentage of the unique strings are valid. It also returns a list of the unique and valid smiles
    """

    n_tot = len(smiles)
    smiles = list(set(smiles))
    n_unique = len(smiles)

    valid_smiles = []
    
    for smile in smiles:
        mol = MolFromSmiles(smile)
        if not isinstance(mol, type(None)):
            valid_smiles.append(smile)
    
    n_valid = len(valid_smiles)

    perc_unique = n_unique/n_tot *100
    perc_valid = n_valid/n_unique *100

    return valid_smiles, perc_unique, perc_valid


def write_smiles(smiles, filename="smiles.smi"):
    """
    This writes some smiles strings to a file.
    """

    f_out = open(filename, "w")

    for i,smile in enumerate(smiles):
        f_out.write(smile + "\t" + str(i) + "\n")

    f_out.close()


