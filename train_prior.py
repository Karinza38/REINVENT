#!/usr/bin/env python

import torch
from torch.utils.data import DataLoader
import pickle
from rdkit import Chem
from rdkit import rdBase
from tqdm import tqdm

from data_structs import MolData, Vocabulary
from model import RNN
from utils import Variable, decrease_learning_rate
rdBase.DisableLog('rdApp.error')

def pretrain(restore_from=None, save_to="data/Prior.ckpt", data="data/mols_filtered.smi", voc_file="data/Voc", batch_size=128, learning_rate=0.001, n_epochs=5, store_loss_dir=None):
    """Trains the Prior RNN"""

    # Read vocabulary from a file
    voc = Vocabulary(init_from_file=voc_file)

    # Create a Dataset from a SMILES file
    moldata = MolData(data, voc)
    data = DataLoader(moldata, batch_size=batch_size, shuffle=True, drop_last=True,
                      collate_fn=MolData.collate_fn)

    Prior = RNN(voc, batch_size)

    # Adding a file to log loss info
    if store_loss_dir is None:
        out_f = open("loss.csv", "w")
    else:
        out_f = open("{}/loss.csv".format(store_loss_dir.rstrip("/")), "w")

    out_f.write("Step,Loss\n")

    # Can restore from a saved RNN
    if restore_from:
        Prior.rnn.load_state_dict(torch.load(restore_from))
    
    # For later plotting the loss
    training_step_counter = 0
    n_logging = 100

    optimizer = torch.optim.Adam(Prior.rnn.parameters(), lr = learning_rate)
    for epoch in range(1, n_epochs+1):
        # When training on a few million compounds, this model converges
        # in a few of epochs or even faster. If model sized is increased
        # its probably a good idea to check loss against an external set of
        # validation SMILES to make sure we dont overfit too much.
        for step, batch in tqdm(enumerate(data), total=len(data)):

            # Sample from DataLoader
            seqs = batch.long()

            # Calculate loss
            log_p, _ = Prior.likelihood(seqs)
            loss = - log_p.mean()

            # Calculate gradients and take a step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Logging the loss to a file
            if training_step_counter % n_logging == 0:
                out_f.write("{},{}\n".format(step,loss))
                training_step_counter += 1

            # Every 500 steps we decrease learning rate and print some information
            if step % 500 == 0 and step != 0:
                decrease_learning_rate(optimizer, decrease_by=0.03)
                tqdm.write("*" * 50)
                tqdm.write("Epoch {:3d}   step {:3d}    loss: {:5.2f}\n".format(epoch, step, loss.data))
                seqs, likelihood, _ = Prior.sample(128)
                valid = 0
                for i, seq in enumerate(seqs.cpu().numpy()):
                    smile = voc.decode(seq)
                    if Chem.MolFromSmiles(smile):
                        valid += 1
                    if i < 5:
                        tqdm.write(smile)
                tqdm.write("\n{:>4.1f}% valid SMILES".format(100 * valid / len(seqs)))
                tqdm.write("*" * 50 + "\n")
                torch.save(Prior.rnn.state_dict(), save_to)

        # Save the Prior
        torch.save(Prior.rnn.state_dict(), save_to)

    f_out.close()


if __name__ == "__main__":
    pretrain(save_to="../models/Prior_chembl_p2x7.ckpt", data="../datasets/filtered/chembl23_training_p2x7.smi", voc_file="../vocabularies/Voc_joined", batch_size=64)
