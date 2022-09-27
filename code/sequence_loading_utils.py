### Loading libraries

## Classic libraries
import os
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from collections import Counter

## Pytorch libraries
import torch
import torch.nn as nn
from torch.utils import data
import torch.nn.functional as F
from torchvision import transforms
import torchvision.models as models
from torchsummary import summary

# Torchtext Libraries
from torchtext.vocab import vocab

# Pytorch Data
import torchdata

# Torch optimizer
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim import Adam

def IMDB_vocab_constructor(tokenizer_fnc = 'basic_english', 
    unknown = '<unk>', beginning = '<BOS>', 
    ending = '<EOS>', padding = '<PAD>'):

    '''
    Imports the IMDB dataset library, performs tokenization, 
    and constructs

    Input: 
    - tokenizer funciton 
    (see https://pytorch.org/text/stable/data_utils.html)

    Returns:
    - the vocab list
    - length of vocab list
    - the maximum line length of each entry
    (for positional encoding)
    - 
    '''

    import torchtext
    from torchtext.datasets import IMDB
    from torchtext.data.utils import get_tokenizer

    # Loading Data
    train_iter, test_iter = IMDB(split=('train', 'test'))

    # Tokenizing the inputs
    tokenizer = get_tokenizer(tokenizer_fnc)


    ## Creating a vocab list

    counter = Counter() # Starting a counter for vocab
    max_line_length = 0; # For positional encoding

    for (label, line) in iter(train_iter):
        tokenized_line = tokenizer(line);
        counter.update(tokenized_line)

        if len(tokenized_line) > max_line_length:

            # The maximum line length is the length of the tokenized line plus 2 to account for <EOS> and <BOS> 
            max_line_length = len(tokenized_line) + 2


    # Creating special characters
    special_characters = (unknown, beginning, ending, padding);
    vocab_list = vocab(counter, min_freq=10, specials=special_characters)
    vocab_list.set_default_index(vocab_list[unknown])

    # Determining vocab length
    vocab_length = len(vocab_list)

    collate_batch_fn = batch_collator(vocab_list, tokenizer)

    return(vocab_list, vocab_length, max_line_length, collate_batch_fn)



def sentence_tokenizer(sentence, vocab_list, tokenizer, bos = '<BOS>', eos = '<EOS>'):
    '''
    Tokenizes an input sentence.

    Inputs:
    - sentence
    - vocab_list
    - bos: <BOS> token (set to None if not using)
    - eos: <EOS> token (set to None if not using)
    - tokenizer: should be a tokenizer function (e.g., get_tokenizer('basic_english'))

    Returns: 
    - tokenized sentence
    '''

    tokenized_sentence = [vocab_list[token] for token in tokenizer(sentence)]

    if bos is not None:
        tokenized_sentence = [vocab_list[bos]] + tokenized_sentence

    if eos is not None:
        tokenized_sentence = tokenized_sentence + [vocab_list[eos]]

    return(tokenized_sentence)

def label_transform(label, pos_flag = 'pos'):
    '''
    Converts labels to 1 or 0
    '''

    return(1 if label == pos_flag else 0)


def label_transform(label):
    return(1 if label == 'pos' else 0)


class batch_collator(object):
    def __init__(self, vocab_list, tokenizer):

        '''
        Processing object for data.DataLoader that tokenizes the input lines and labels and collates it into batch tensor of the right shape.

        Inputs:
        - vocab_list
        - tokenizer function
        '''

        self.vocab_list = vocab_list;
        self.tokenizer = tokenizer; 

    def __call__(self, batch):

        '''
        When passed, returns a tokenized batch
        '''

        import torch
        from torch.nn.utils.rnn import pad_sequence

        processed_labels = []
        processed_lines = []

        for batch_label, batch_line in batch: 
        
            processed_labels.append(label_transform(batch_label));
            processed_lines.append(torch.tensor(sentence_tokenizer(batch_line, vocab_list = self.vocab_list, tokenizer = self.tokenizer)));
        
        # Creates padding across the batch
        processed_lines = pad_sequence(processed_lines, padding_value = self.vocab_list['<PAD>']);
        
        # Making sure the shape is batch-first 
        if processed_lines.shape[0] is not len(batch):
            processed_lines = processed_lines.T

        return(torch.tensor(processed_labels), processed_lines)
        