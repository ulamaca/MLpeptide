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
folder = "/data/AIpep-clean/"
import matplotlib.pyplot as plt
from vocabulary import Vocabulary
from datasetbioactivity import Dataset
from datasetbioactivity import collate_fn_no_activity as collate_fn
from models import Generator
from tqdm.autonotebook  import trange, tqdm
import os
from collections import defaultdict

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


if __name__ == "__main__":
    # TODO: this requires pytorch=1.5.0
    # TODO: wanted to test the training runtime for a minimal spec machine
    
    # MODEL HYPERPARAMS
    n_embedding  = 100
    n_hidden = 400
    n_layers = 2
    n_epoch = 150
    learning_rate = 0.001
    momentum = 0.9
    batch_size = 10


    #df = pd.read_pickle(folder + "pickles/DAASP_RNN_dataset.plk")
    df = pd.read_csv('data/DAASP_RNN_dataset.csv')

    df_training = df[df["Set"]=="training"]
    df_test = df[df["Set"]=="test"]

    vocabulary = Vocabulary.get_vocabulary_from_sequences(df_training.Sequence.values)

    if torch.cuda.is_available():
        device = "cuda" 
    else:
        device = "cpu" 
    
    if not os.path.exists(folder+"pickles/generator_training_results.pkl"):
        df_training_active = df_training.query("activity == 1")
        df_test_active = df_test.query("activity == 1")
        df_training_inactive = df_training.query("activity == 0")
        df_test_inactive = df_test.query("activity == 0")

        training_dataset_active = Dataset(df_training_active, vocabulary, with_activity=False)
        test_dataset_active = Dataset(df_test_active, vocabulary, with_activity=False)
        training_dataset_inactive = Dataset(df_training_inactive, vocabulary, with_activity=False)
        test_dataset_inactive = Dataset(df_test_inactive, vocabulary, with_activity=False)

        model = Generator(n_embedding, n_hidden, n_layers, vocabulary)
        model.to(device)

        optimizer = optim.SGD(model.model.parameters(), lr = learning_rate, momentum=momentum)

        # the only one used for training
        training_dataloader_active = torch.utils.data.DataLoader(training_dataset_active, batch_size=batch_size, shuffle=True, collate_fn = collate_fn, drop_last=True, pin_memory=True, num_workers=4)

        # used for evaluation
        test_dataloader_active = torch.utils.data.DataLoader(test_dataset_active, batch_size=batch_size, shuffle=False, collate_fn = collate_fn, drop_last=False, pin_memory=True, num_workers=4)
        training_dataloader_inactive = torch.utils.data.DataLoader(training_dataset_inactive, batch_size=batch_size, shuffle=False, collate_fn = collate_fn, drop_last=False, pin_memory=True, num_workers=4)
        test_dataloader_inactive = torch.utils.data.DataLoader(test_dataset_inactive, batch_size=batch_size, shuffle=False, collate_fn = collate_fn, drop_last=False, pin_memory=True, num_workers=4)
        training_dataloader_active_eval = torch.utils.data.DataLoader(training_dataset_active, batch_size=batch_size, shuffle=False, collate_fn = collate_fn, drop_last=False, pin_memory=True, num_workers=4)

        training_dictionary = {}

        for e in trange(1, n_epoch + 1):
            print("Epoch {}".format(e))
            for i_batch, sample_batched in tqdm(enumerate(training_dataloader_active), total=len(training_dataloader_active) ):

                seq_batched = sample_batched[0].to(model.device, non_blocking=True) 
                seq_lengths = sample_batched[1].to(model.device, non_blocking=True)

                nll = model.likelihood(seq_batched, seq_lengths)

                loss = nll.mean()

                optimizer.zero_grad()
                loss.backward()  
                torch.nn.utils.clip_grad_value_(model.model.parameters(), 2)
                optimizer.step()

            model.save(folder+"models/RNN-generator/ep{}.pkl".format(e))


            print("\tExample Sequences")
            sampled_seq = model.sample(5)
            for s in sampled_seq:
                print("\t\t{}".format(model.vocabulary.tensor_to_seq(s, debug=True)))

            nll_training = []
            with torch.no_grad():
                for i_batch, sample_batched in enumerate(training_dataloader_active_eval):    
                    seq_batched = sample_batched[0].to(model.device, non_blocking=True) 
                    seq_lengths = sample_batched[1].to(model.device, non_blocking=True) 

                    nll_training += model.likelihood(seq_batched, seq_lengths)

            nll_training_active_mean = torch.stack(nll_training).mean().item()
            print("\tNLL Train Active: {}".format(nll_training_active_mean))
            del nll_training

            nll_test = []
            with torch.no_grad():
                for i_batch, sample_batched in enumerate(test_dataloader_active):    
                    seq_batched = sample_batched[0].to(model.device, non_blocking=True) 
                    seq_lengths = sample_batched[1].to(model.device, non_blocking=True) 

                    nll_test += model.likelihood(seq_batched, seq_lengths)

            nll_test_active_mean = torch.stack(nll_test).mean().item()
            print("\tNLL Test Active: {}".format(nll_test_active_mean))
            del nll_test

            nll_training = []
            with torch.no_grad():
                for i_batch, sample_batched in enumerate(training_dataloader_inactive):    
                    seq_batched = sample_batched[0].to(model.device, non_blocking=True) 
                    seq_lengths = sample_batched[1].to(model.device, non_blocking=True) 

                    nll_training += model.likelihood(seq_batched, seq_lengths)

            nll_training_inactive_mean = torch.stack(nll_training).mean().item()
            print("\tNLL Train Inactive: {}".format(nll_training_inactive_mean))
            del nll_training

            nll_test = []
            with torch.no_grad():
                for i_batch, sample_batched in enumerate(test_dataloader_inactive):    
                    seq_batched = sample_batched[0].to(model.device, non_blocking=True) 
                    seq_lengths = sample_batched[1].to(model.device, non_blocking=True) 

                    nll_test += model.likelihood(seq_batched, seq_lengths)

            nll_test_inactive_mean = torch.stack(nll_test).mean().item()
            print("\tNLL Test Inactive: {}".format(nll_test_inactive_mean))
            del nll_test
            print()

            training_dictionary[e]=[nll_training_active_mean, nll_test_active_mean, nll_training_inactive_mean, nll_test_inactive_mean]
        
        with open(folder+"pickles/generator_training_results.pkl",'wb') as fd:
            pickle.dump(training_dictionary, fd)
        
    else:
        with open(folder+"pickles/generator_training_results.pkl",'rb') as fd:
            training_dictionary = pickle.load(fd)

    min_nll_test_active = float("inf")
    for epoch, training_values in training_dictionary.items():
        nll_test_active = training_values[1]

        if nll_test_active < min_nll_test_active:
            best_epoch = epoch
            min_nll_test_active = nll_test_active

        
