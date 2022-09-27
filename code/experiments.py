### Loading libraries

# Classic libraries
import os
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from collections import Counter
import math

# Pytorch libraries
import torch
import torch.nn as nn
from torch.utils import data
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import torchvision.models as models
from torchsummary import summary
import torchdata
import torchtext
from torchmetrics import Accuracy

# Torch optimizer
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim import Adam

# Custom libraries
from classification_transformer_utils import *
from sequence_loading_utils import * 
from classification_encoders import * 

### Loading Data
from torchtext.datasets import IMDB

print("Libraries successfully imported")

train_data, test_data = list(IMDB(split=('train','test')))

train_class = [int(label == 'pos') for (label, line) in iter(train_data)]
test_class = [int(label == 'pos') for (label, line) in iter(test_data)]

print("Positive reviews in training dataset: ",sum(train_class)/len(train_class))
print("Positive reviews in test dataset: ",sum(test_class)/len(test_class))

# Tokenizing and creating Vocab list
vocab_list, vocab_length, max_line_length, collate_batch_fn = IMDB_vocab_constructor();

train_loader = data.DataLoader(list(IMDB(split=('train'))), batch_size = 32, shuffle=True, collate_fn = collate_batch_fn);
test_loader = data.DataLoader(list(IMDB(split=('test'))), batch_size = 32, shuffle=True, collate_fn = collate_batch_fn);

### Training Models

## Baseline Classification Transformer

# Hyperparameters
model = ClassificationEncoder(embedding_dim = 50, encoder_stack = 6,
                              num_heads = 5, layer_norm_epsilon = 1e-6, 
                              dropout_rate = 0.1, input_vocab_size = vocab_length, 
                              number_of_classes = 2, max_positions = max_line_length)
                              

criterion = nn.CrossEntropyLoss()

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), 
                              betas = (0.9, 0.98), eps = 1.0e-9, lr = 1E-4)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


[model, epoch_training_loss_array,
 epoch_test_loss_array, epoch_training_accuracy_array, 
 epoch_test_accuracy_array, 
 time_for_epochs, _, _] = train_and_evaluate(model, 
                                             train_loader, 
                                             test_loader, 
                                             optimizer, 
                                             criterion,
                                             device, 
                                             scheduler = None, 
                                             epochs = 100, 
                                             verbose = True, 
                                             record_lr = False);

save_outputs(model, epoch_training_loss_array, epoch_test_loss_array, 
             epoch_training_accuracy_array, epoch_test_accuracy_array, 
             time_for_epochs, model_filename = 'regular_e5_classTransformer.ckpt', 
             data_filename = 'data_regular_e5_classTransformer.csv');


## Classification Transformer with Key Padding Masks
def create_key_padding_mask(lines,padding_value):
  # Creating key padding mask
    return(lines == padding_value);

model = ClassificationEncoder(embedding_dim = 50, encoder_stack = 6,
                              num_heads = 5, layer_norm_epsilon = 1e-6, 
                              dropout_rate = 0.1, input_vocab_size = vocab_length, 
                              number_of_classes = 2, max_positions = max_line_length,
                              padding_function = create_key_padding_mask, 
                              padding_value = vocab_list['<unk>'])
                              
criterion = nn.CrossEntropyLoss()

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), 
                              betas = (0.9, 0.98), eps = 1.0e-9, lr = 1E-4)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

[model, epoch_training_loss_array,
 epoch_test_loss_array, epoch_training_accuracy_array, 
 epoch_test_accuracy_array, 
 time_for_epochs, _, _]  = train_and_evaluate(model, 
                                             train_loader, 
                                             test_loader, 
                                             optimizer, 
                                             criterion,
                                             device, 
                                             scheduler = None, 
                                             epochs = 100, 
                                             verbose = True, 
                                             record_lr = False);

save_outputs(model, epoch_training_loss_array, epoch_test_loss_array, 
             epoch_training_accuracy_array, epoch_test_accuracy_array, 
             time_for_epochs,model_filename = 'kpv_e100_classTransformer.ckpt', 
             data_filename = 'data_kpv_e100_classTransformer.csv');

## Classification Transformer with Key Padding Masks and Dropout Layers

model = ClassificationEncoder_dropout(embedding_dim = 50, encoder_stack = 6,
                              num_heads = 5, layer_norm_epsilon = 1e-6, 
                              dropout_rate = 0.1, input_vocab_size = vocab_length, 
                              number_of_classes = 2, max_positions = max_line_length,
                              padding_function = create_key_padding_mask, 
                              padding_value = vocab_list['<unk>'])
                              

criterion = nn.CrossEntropyLoss()

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), 
                              betas = (0.9, 0.98), eps = 1.0e-9, lr = 1E-4)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

[model, epoch_training_loss_array,
 epoch_test_loss_array, epoch_training_accuracy_array, 
 epoch_test_accuracy_array, 
 time_for_epochs, _, _]  = train_and_evaluate(model, 
                                             train_loader, 
                                             test_loader, 
                                             optimizer, 
                                             criterion,
                                             device, 
                                             scheduler = None, 
                                             epochs = 100, 
                                             verbose = True, 
                                             record_lr = False);

save_outputs(model, epoch_training_loss_array, epoch_test_loss_array, 
             epoch_training_accuracy_array, epoch_test_accuracy_array, 
             time_for_epochs, model_filename = 'dropout_e100_classTransformer.ckpt', 
             data_filename = 'dropout_kpv_e100_classTransformer.csv');

### Data Visualization
datasets = [['Baseline model','data_regular_e5_classTransformer.csv','lightblue'],
            ['With key padding values (KPV)','data_kpv_e100_classTransformer.csv','blue'],
            ['With key padding values (KPV) + dropout','dropout_kpv_e100_classTransformer.csv','teal']]

# Visualizing Accuracy

legend = [];

plt.rcParams["figure.figsize"] = (12,7)
plt.rcParams['font.size'] = '14'

for label, csvfile, linecolor  in datasets:

  model_data = pd.read_csv(csvfile)
  plt.plot(model_data['Training Accuracy'][0:100],'-',color=linecolor)
  plt.plot(model_data['Test Accuracy'][0:100],'--',color=linecolor)
  legend.append(label+', Training')
  legend.append(label+', Test Set')

plt.legend(legend)
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.ylim([0,1])

# Visualizing Loss

legend = [];

plt.rcParams["figure.figsize"] = (12,7)
plt.rcParams['font.size'] = '14'

for label, csvfile, linecolor  in datasets:

  model_data = pd.read_csv(csvfile)
  plt.plot(model_data['Training Loss'][0:100],'-',color=linecolor)
  plt.plot(model_data['Test Loss'][0:100],'--',color=linecolor)
  legend.append(label+', Training')
  legend.append(label+', Test Set')

plt.legend(legend,loc=0)
plt.ylabel('Loss')
plt.xlabel('Epochs')


