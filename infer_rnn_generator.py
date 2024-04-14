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
#from datasetbioactivity import collate_fn
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

def uniqueness(seqs):
    unique_seqs = defaultdict(int)
    for s in seqs:
        unique_seqs[s] += 1
    return unique_seqs, (len(unique_seqs)/len(seqs))*100

## for predictive models
def predict_wo_label(data_loader, model):    
    out_list = []

    with torch.no_grad():
        for i_batch, sample_batched in enumerate(data_loader):                
            
            seq_batched = sample_batched[0].to(model.device, non_blocking=True)
            seq_lengths = sample_batched[1].to(model.device, non_blocking=True)            
            out_list += torch.exp(model.evaluate(seq_batched, seq_lengths))[: ,1].to("cpu", non_blocking=True)
        
        out_list = torch.stack(out_list)
    return out_list.cpu().numpy()

def load_model():
    from models import Classifier
    filename = 'data/AIpep-clean/models/RNN-classifier/em100_hi400_la2_ep48'
    model = Classifier.load_from_file(filename)    

    return model

if __name__ == "__main__":
    
    # Generative Models
    n_samples = 1000
    best_epoch = 23 # from the original MLpeptide notebook
    model_path = 'data/AIpep-clean/archive/v2-gpu-20240405/RNN-generator'
    model = Generator.load_from_file(os.path.join(model_path, f"ep{best_epoch}.pkl"))

    sequence = 'TKPRPGP' # peptide: Selank
    sequences = [sequence]
    
    
    # TODO, for development of gumbel ST learner
    #model.yay(sequences)
    seqs = _sample(model, n_samples)
    unique_seqs, perc_uniqueness = uniqueness(seqs)
    print(f"uniqueness: {perc_uniqueness}")
    print("generated sequences:")
    for seq in unique_seqs:
        print(seq)    


    ## CLS model
    clf = load_model()
    from datasetbioactivity import Dataset
    df = pd.DataFrame([{'Sequence': seq} for seq in unique_seqs])    
    gen_dataset = Dataset(df, clf.vocabulary, with_activity=False)    
    gen_dataloader = torch.utils.data.DataLoader(gen_dataset, batch_size=1, shuffle=False, collate_fn = collate_fn, drop_last=True, pin_memory=True, num_workers=4)
    y_score = predict_wo_label(gen_dataloader, clf)
    breakpoint()