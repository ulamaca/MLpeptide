import random
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
from config import folder
import matplotlib.pyplot as plt
from vocabulary import Vocabulary
from datasetbioactivity import Dataset, collate_fn
from models import Classifier
import os

## Helpers
def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]

def categoryFromOutput(output):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return category_i

def nan_equal(a,b):
    try:
        np.testing.assert_equal(a,b)
    except AssertionError:
        return False
    return True

def models_are_equal(model1, model2):
    model1.vocabulary == model2.vocabulary
    model1.hidden_size == model2.hidden_size
    for a,b in zip(model1.model.parameters(), model2.model.parameters()):
        if nan_equal(a.detach().numpy(), b.detach().numpy()) == True:
            print("true")

# MAINs
def training(model, test_dataloader, training_dataloader, n_epoch, optimizer, filename):
    
    roc_training = []
    roc_test = []
    
    for e in range(1, n_epoch + 1):
        for i_batch, sample_batched in enumerate(training_dataloader):
            seq_batched = sample_batched[0][0].to(model.device, non_blocking=True)
            seq_lengths = sample_batched[0][1].to(model.device, non_blocking=True)
            cat_batched = sample_batched[1].to(model.device, non_blocking=True)

            output = model.evaluate(seq_batched, seq_lengths)

            loss = criterion(output, cat_batched)

            optimizer.zero_grad()
            loss.backward()  
            torch.nn.utils.clip_grad_value_(model.model.parameters(), 2)
            optimizer.step()

        model.save(filename.format(e))
        
        def _evaluate_ROC(data_loader):
            cat_list = []
            out_list = []

            with torch.no_grad():
                for i_batch, sample_batched in enumerate(data_loader):    
                    seq_batched = sample_batched[0][0].to(model.device, non_blocking=True)
                    seq_lengths = sample_batched[0][1].to(model.device, non_blocking=True)
                    
                    cat_list += sample_batched[1].to("cpu", non_blocking=True)
                    out_list += torch.exp(model.evaluate(seq_batched, seq_lengths))[: ,1].to("cpu", non_blocking=True)

                cat_list = torch.stack(cat_list)
                out_list = torch.stack(out_list)

                roc = roc_auc_score(cat_list.cpu().numpy().astype(int), out_list.cpu().numpy())
            return roc
        
        roc_tr = _evaluate_ROC(training_dataloader)
        roc_te = _evaluate_ROC(test_dataloader)
        roc_training.append(roc_tr)
        roc_test.append(roc_te)
        print("epoch: " + str(e))
        print("roc auc training: " + str(roc_tr))
        print("roc auc test: " + str(roc_te))
        if roc_training == 1.0:
            break
        
    return model, optimizer, roc_training, roc_test

# Hyperparms
learning_rate = 0.01
momentum = 0.9
batch_size = 20
n_epoch = 150
criterion = nn.NLLLoss()
n_hidden = 400
n_embedding = 100
n_layers = 2



if __name__ == "__main__":
    df = pd.read_csv('data/DAASP_RNN_dataset.csv')

    df_training = df[df["Set"]=="training"]
    df_test = df[df["Set"]=="test"]

    vocabulary = Vocabulary.get_vocabulary_from_sequences(df_training.Sequence.values)

    if torch.cuda.is_available():
        device = "cuda" 
    else:
        device = "cpu" 
    
    training_dataset = Dataset(df_training, vocabulary)
    test_dataset = Dataset(df_test, vocabulary)

    print(f"dimensions of embedding {n_embedding}, dimensions of hidden {n_hidden}, number of layers {n_layers}")

    model = Classifier(n_embedding, n_hidden, n_layers, vocabulary)
    model.to(device)

    optimizer = optim.SGD(model.model.parameters(), lr = learning_rate, momentum=momentum)
    training_dataloader = torch.utils.data.DataLoader(training_dataset, batch_size=batch_size, shuffle=True, collate_fn = collate_fn, drop_last=True, pin_memory=True, num_workers=4)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn = collate_fn, drop_last=True, pin_memory=True, num_workers=4)

    filename = folder + "/models/RNN-classifier/em{}_hi{}_la{}_ep{{}}".format(n_embedding, n_hidden, n_layers)
    model, optimizer, roc_training, roc_test = training(model, test_dataloader, training_dataloader, n_epoch, optimizer, filename)

    print(f"maximum roc auc for test set {max(roc_test)}")