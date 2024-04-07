import random
import os
import torch.nn as nn
import torch
import pickle
import pandas as pd
from pandas import Series, DataFrame
from pandarallel import pandarallel
pandarallel.initialize(progress_bar=False)
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, matthews_corrcoef, f1_score, precision_score, recall_score
import numpy as np
import torch.optim as optim
folder = "data/AIpep-clean/"
import matplotlib.pyplot as plt
from vocabulary import Vocabulary
from datasetbioactivity import Dataset
from datasetbioactivity import collate_fn_no_activity as collate_fn
from models import Generator
from tqdm.autonotebook  import trange, tqdm
import os
from collections import defaultdict

# from the original MLpeptide notebook
def _sample(model, n):
    sampled_seq = model.sample(n)
    sequences = []
    for s in sampled_seq:
        sequences.append(model.vocabulary.tensor_to_seq(s))
    return sequences

if __name__ == "__main__":
    best_epoch = 23 # from the original MLpeptide notebook
    model_path = 'data/AIpep-clean/archive/v2-gpu-20240405/RNN-generator'
    model = Generator.load_from_file(os.path.join(model_path, f"ep{best_epoch}.pkl"))

    sequence = 'TKPRPGP' # peptide: Selank
    sequences = [sequence]
    
    # TODO, for development
    model.yay(sequences)
    _sample(model, 5)
    breakpoint()