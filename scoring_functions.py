#!/usr/bin/env python
from __future__ import print_function, division
import numpy as np
from rdkit import Chem
from rdkit import rdBase
from rdkit.Chem import AllChem, Descriptors
from rdkit import DataStructs
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
from sklearn import svm
import keras
import time
import pickle
import re
import threading
import pexpect
import sys
import os
rdBase.DisableLog('rdApp.error')

# Needed for the synthetisability score
from SA_score import sascorer

"""Scoring function should be a class where some tasks that are shared for every call
   can be reallocated to the __init__, and has a __call__ method which takes a single SMILES of
   argument and returns a float. A multiprocessing class will then spawn workers and divide the
   list of SMILES given between them.

   Passing *args and **kwargs through a subprocess call is slightly tricky because we need to know
   their types - everything will be a string once we have passed it. Therefor, we instead use class
   attributes which we can modify in place before any subprocess is created. Any **kwarg left over in
   the call to get_scoring_function will be checked against a list of (allowed) kwargs for the class
   and if a match is found the value of the item will be the new value for the class.

   If num_processes == 0, the scoring function will be run in the main process. Depending on how
   demanding the scoring function is and how well the OS handles the multiprocessing, this might
   be faster than multiprocessing in some cases."""

class no_sulphur():
    """Scores structures based on not containing sulphur."""

    kwargs = []

    def __init__(self):
        pass
    def __call__(self, smile):
        mol = Chem.MolFromSmiles(smile)
        if mol:
            has_sulphur = [16 not in [atom.GetAtomicNum() for atom in mol.GetAtoms()]]
            return float(has_sulphur)
        return 0.0

class tanimoto():
    """Scores structures based on Tanimoto similarity to a query structure.
       Scores are only scaled up to k=(0,1), after which no more reward is given."""

    kwargs = ["k", "query_structure"]
    k = 0.7
    query_structure = "Cc1ccc(cc1)c2cc(nn2c3ccc(cc3)S(=O)(=O)N)C(F)(F)F"

    def __init__(self):
        query_mol = Chem.MolFromSmiles(self.query_structure)
        self.query_fp = AllChem.GetMorganFingerprint(query_mol, 2, useCounts=True, useFeatures=True)

    def __call__(self, smile):
        mol = Chem.MolFromSmiles(smile)
        if mol:
            fp = AllChem.GetMorganFingerprint(mol, 2, useCounts=True, useFeatures=True)
            score = DataStructs.TanimotoSimilarity(self.query_fp, fp)
            score = min(score, self.k) / self.k
            return float(score)
        return 0.0

class activity_model():
    """Scores based on an ECFP classifier for activity."""

    kwargs = ["clf_path"]
    clf_path = 'data/clf.pkl'

    def __init__(self):
        with open(self.clf_path, "rb") as f:
            self.clf = pickle.load(f)

    def __call__(self, smile):
        mol = Chem.MolFromSmiles(smile)
        if mol:
            fp = activity_model.fingerprints_from_mol(mol)
            score = self.clf.predict_proba(fp)[:, 1]
            return float(score)
        return 0.0

    @classmethod
    def fingerprints_from_mol(cls, mol):
        fp = AllChem.GetMorganFingerprint(mol, 3, useCounts=True, useFeatures=True)
        size = 2048
        nfp = np.zeros((1, size), np.int32)
        for idx,v in fp.GetNonzeroElements().items():
            nidx = idx%size
            nfp[0, nidx] += int(v)
        return nfp

class pIC50_pred():
    """Scores based on an MFP classifier for activity."""

    kwargs = ['path_to_model', 'path_to_scaler', 'pic50_term']

    def __init__(self):

        self.clf = keras.models.load_model(self.path_to_model)
        if self.path_to_scaler == '':
            self.scaler = None
        else:
            self.scaler = pickle.load(open(self.path_to_scaler, 'rb'))

        if isinstance(self.pic50_term, type(None)):
            self.pic50_term = 7

    def __call__(self, smile):

        mol = Chem.MolFromSmiles(smile)
        if mol:
            fp = GetMorganFingerprintAsBitVect(mol, radius=3, nBits=2048)
            fp_arr = np.zeros((1,))
            DataStructs.ConvertToNumpyArray(fp, fp_arr)
            if not isinstance(self.scaler, type(None)):
                scaled_fp = self.scaler.transform(np.expand_dims(fp_arr, 0))
            else:
                scaled_fp = np.expand_dims(fp_arr, 0)
            pic50 = self.clf.predict(scaled_fp)
            score = np.tanh(pic50-self.pic50_term)
            return score
        return -1.0

class pIC50_mw():
    """Scores based on an MFP classifier for activity and RDKit for Molecular weight."""

    kwargs = ['path_to_model', 'path_to_scaler', 'pic50_term', 'mw_term', 'std_term']

    def __init__(self):

        self.clf = keras.models.load_model(self.path_to_model)
        if self.path_to_scaler == '':
            self.scaler = None
        else:
            self.scaler = pickle.load(open(self.path_to_scaler, 'rb'))

        if isinstance(self.pic50_term, type(None)):
            self.pic50_term = 7

        if isinstance(self.mw_term, type(None)):
            self.mw_term = 395

        if isinstance(self.std_term, type(None)):
            self.std_term = 70

    def __call__(self, smile):

        mol = Chem.MolFromSmiles(smile)
        if not isinstance(mol, type(None)):
            mw = Descriptors.ExactMolWt(mol) - self.mw_term
            fp = GetMorganFingerprintAsBitVect(mol, radius=3, nBits=2048)
            fp_arr = np.zeros((1,))
            DataStructs.ConvertToNumpyArray(fp, fp_arr)
            if not isinstance(self.scaler, type(None)):
                scaled_fp = self.scaler.transform(np.expand_dims(fp_arr, 0))
            else:
                scaled_fp = np.expand_dims(fp_arr, 0)
            pic50 = self.clf.predict(scaled_fp)
            #score = 0.5*np.tanh(pic50-7) + 0.5*(2*np.exp((-mw**2)/2*self.std_term**2)-1)
            score = 0.5*np.tanh(pic50-7) + 0.5*(2*np.exp(-(0.009*mw)**2)-1)
            return score
        return -1.0

class pIC50_synth():
    """ Scores based on a MFP classifier for activity and a synthetisability score"""
    
    kwargs = ['path_to_model', 'path_to_scaler', 'pic50_term']

    def __init__(self):

        # Loading the pIC50 model
        self.clf = keras.models.load_model(self.path_to_model)
        if self.path_to_scaler == '':
            self.scaler = None
        else:
            self.scaler = pickle.load(open(self.path_to_scaler, 'rb'))

        if isinstance(self.pic50_term, type(None)):
            self.pic50_term = 7
   
        # Loading the Synthetic Complexity scorer
        #self.scscorer = SCScorer()
        #self.scscorer.restore(weight_path="../../scscore/models/full_reaxys_model_1024bool/model.ckpt-10654.as_numpy.json.gz")

    def __call__(self, smile):

        mol = Chem.MolFromSmiles(smile)
        if mol:
            fp = GetMorganFingerprintAsBitVect(mol, radius=3, nBits=2048)
            fp_arr = np.zeros((1,))
            DataStructs.ConvertToNumpyArray(fp, fp_arr)
            if not isinstance(self.scaler, type(None)):
                scaled_fp = self.scaler.transform(np.expand_dims(fp_arr, 0))
            else:
                scaled_fp = np.expand_dims(fp_arr, 0)
            pic50 = self.clf.predict(scaled_fp)

            # Obtaining the synthetic complexity score for the smile
            #_, sc_score = self.scscorer.get_score_from_smi(smile)

            # Obtaining the synthetic accessibility score for the smile
            sa_score = sascorer.calculateScore(mol)

            # Obtaining the average of the pIC50 and sa score
            score = 0.5*np.tanh(pic50-self.pic50_term) + 0.5*((-sa_score+5)*0.5-1)
            return score
        return -1.0

class pIC50_mw_synth():
    """Scores based on an MFP classifier for activity and RDKit for Molecular weight and Synthetic Accessibility."""

    kwargs = ['path_to_model', 'path_to_scaler', 'pic50_term', 'mw_term', 'std_term']

    def __init__(self):

        self.clf = keras.models.load_model(self.path_to_model)
        if self.path_to_scaler == '':
            self.scaler = None
        else:
            self.scaler = pickle.load(open(self.path_to_scaler, 'rb'))

        if isinstance(self.pic50_term, type(None)):
            self.pic50_term = 7

        if isinstance(self.mw_term, type(None)):
            self.mw_term = 395

        if isinstance(self.std_term, type(None)):
            self.std_term = 70
    def __call__(self, smile):

        mol = Chem.MolFromSmiles(smile)
        if not isinstance(mol, type(None)):
            mw = Descriptors.ExactMolWt(mol) - self.mw_term
            fp = GetMorganFingerprintAsBitVect(mol, radius=3, nBits=2048)
            fp_arr = np.zeros((1,))
            DataStructs.ConvertToNumpyArray(fp, fp_arr)
            if not isinstance(self.scaler, type(None)):
                scaled_fp = self.scaler.transform(np.expand_dims(fp_arr, 0))
            else:
                scaled_fp = np.expand_dims(fp_arr, 0)
            pic50 = self.clf.predict(scaled_fp)
 
            # Obtaining the synthetic accessibility score for the smile
            sa_score = sascorer.calculateScore(mol)

            # Averaging all the scores
            #score = (np.tanh(pic50-7) + (2*np.exp((-mw**2)/2*self.std_term**2)-1) + ((-sa_score+5)*0.5-1))/3
            score = (np.tanh(pic50-7) + (2*np.exp(-(0.009*mw)**2)-1) + ((-sa_score+5)*0.5-1))/3
            return score
        return -1.0

class Worker():
    """A worker class for the Multiprocessing functionality. Spawns a subprocess
       that is listening for input SMILES and inserts the score into the given
       index in the given list."""
    def __init__(self, scoring_function=None):
        """The score_re is a regular expression that extracts the score from the
           stdout of the subprocess. This means only scoring functions with range
           0.0-1.0 will work, for other ranges this re has to be modified."""

        self.proc = pexpect.spawn('./multiprocess.py ' + scoring_function,
                                  encoding='utf-8')

        print(self.is_alive())

    def __call__(self, smile, index, result_list):
        self.proc.sendline(smile)
        output = self.proc.expect([re.escape(smile) + " 1\.0+|[0]\.[0-9]+", 'None', pexpect.TIMEOUT])
        if output is 0:
            score = float(self.proc.after.lstrip(smile + " "))
        elif output in [1, 2]:
            score = 0.0
        result_list[index] = score

    def is_alive(self):
        return self.proc.isalive()

class Multiprocessing():
    """Class for handling multiprocessing of scoring functions. OEtoolkits cant be used with
       native multiprocessing (cant be pickled), so instead we spawn threads that create
       subprocesses."""
    def __init__(self, num_processes=None, scoring_function=None):
        self.n = num_processes
        self.workers = [Worker(scoring_function=scoring_function) for _ in range(num_processes)]

    def alive_workers(self):
        return [i for i, worker in enumerate(self.workers) if worker.is_alive()]

    def __call__(self, smiles):
        scores = [0 for _ in range(len(smiles))]
        smiles_copy = [smile for smile in smiles]
        while smiles_copy:
            alive_procs = self.alive_workers()
            if not alive_procs:
               raise RuntimeError("All subprocesses are dead, exiting.")
            # As long as we still have SMILES to score
            used_threads = []
            # Threads name corresponds to the index of the worker, so here
            # we are actually checking which workers are busy
            for t in threading.enumerate():
                # Workers have numbers as names, while the main thread cant
                # be converted to an integer
                try:
                    n = int(t.name)
                    used_threads.append(n)
                except ValueError:
                    continue
            free_threads = [i for i in alive_procs if i not in used_threads]
            for n in free_threads:
                if smiles_copy:
                    # Send SMILES and what index in the result list the score should be inserted at
                    smile = smiles_copy.pop()
                    idx = len(smiles_copy)
                    t = threading.Thread(target=self.workers[n], name=str(n), args=(smile, idx, scores))
                    t.start()
            time.sleep(0.01)
        for t in threading.enumerate():
            try:
                n = int(t.name)
                t.join()
            except ValueError:
                continue
        return np.array(scores, dtype=np.float32)

class Singleprocessing():
    """Adds an option to not spawn new processes for the scoring functions, but rather
       run them in the main process."""
    def __init__(self, scoring_function=None):
        self.scoring_function = scoring_function()
    def __call__(self, smiles):
        scores = [self.scoring_function(smile) for smile in smiles]
        return np.array(scores, dtype=np.float32)

def get_scoring_function(scoring_function, num_processes=None, **kwargs):
    """Function that initializes and returns a scoring function by name"""
    scoring_function_classes = [no_sulphur, tanimoto, activity_model, pIC50_pred, pIC50_mw, pIC50_synth, pIC50_mw_synth]
    scoring_functions = [f.__name__ for f in scoring_function_classes]
    scoring_function_class = [f for f in scoring_function_classes if f.__name__ == scoring_function][0]

    if scoring_function not in scoring_functions:
        raise ValueError("Scoring function must be one of {}".format([f for f in scoring_functions]))

    for k, v in kwargs.items():
        if k in scoring_function_class.kwargs:
            setattr(scoring_function_class, k, v)

    if num_processes == 0:
        return Singleprocessing(scoring_function=scoring_function_class)
    return Multiprocessing(scoring_function=scoring_function, num_processes=num_processes)
