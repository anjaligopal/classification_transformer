## Loading libraries

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



class PositionalEncoding(nn.Module):
  '''
  Calculates sine and cosine positional encodings.
  Based partially on this tutorial, but with no dropout:
  https://pytorch.org/tutorials/beginner/transformer_tutorial.html
  '''

  def __init__(self, embedding_dim, max_positions=5000):
      super().__init__()

      pos_encoding = torch.zeros(max_positions,embedding_dim);
      for i in range(max_positions):
          for j in range(embedding_dim):
              
              if j % 2 == 0:
                  # If the dim is even, use sin 
                  pos_encoding[i,j] = np.sin(i/(10000.0**(j/embedding_dim)));
              else:
                  # If the dim is odd, use cos 
                  pos_encoding[i,j] = np.cos(i/10000.0**((j-1.0)/embedding_dim))


      pos_encoding = pos_encoding.unsqueeze(0)
      self.register_buffer("pos_encoding", pos_encoding)

  def forward(self, x):
        x = x + self.pos_encoding[:, : x.size(1), :]
        return x


class ClassificationEncoder(nn.Module):

    '''
    This encodes for a classification encoder with self attention. 
    '''

    def __init__(self, embedding_dim, num_heads, encoder_stack = 6,
               layer_norm_epsilon = 1e-6, dropout_rate = 0.1, 
               input_vocab_size = 10000, number_of_classes = 2,
               max_positions = 5000, padding_function = None, 
               padding_value = None):

        super().__init__()

        ## Initializing static variables
        self.encoder_stack = encoder_stack; 
        self.embedding_dim = embedding_dim;
        self.padding_value = padding_value;
        self.padding_function = padding_function;

        ## Initializing Layers

        # Positional Encoding
        self.pos_encoding = PositionalEncoding(embedding_dim, max_positions)

        # Word Embedding 
        self.embedding = nn.Embedding(input_vocab_size,embedding_dim)

        # MHA Layer
        self.mha = nn.MultiheadAttention(embed_dim = embedding_dim, 
                                         num_heads = num_heads,
                                         dropout = dropout_rate);

        # Normalization Layer
        self.layer_normalization = nn.LayerNorm(embedding_dim, 
                                                eps=layer_norm_epsilon);
        # Dropout Layers
        self.dropout = nn.Dropout(dropout_rate);

        # FFN Layers with ReLU Activation
        self.linear1 = nn.Linear(embedding_dim, 4*embedding_dim);
        self.relu = nn.ReLU();
        self.linear2 = nn.Linear(4*embedding_dim, embedding_dim);

        # Linear output layer, which connects input sequence to the number
        # of classes
        self.linear_output = nn.Linear(embedding_dim,number_of_classes); 

        # Softmax output layer 
        self.softmax_output = nn.Softmax(dim=-1);

    def forward(self, x):
        '''
        Performs a single forward pass through the encoder
        '''

        # Key padding mask
        if self.padding_function is not None:
          key_pad_mask = self.padding_function(x,padding_value = self.padding_value).T
        else:
          key_pad_mask = None

        # Performs embedding
        x = self.embedding(x) * math.sqrt(self.embedding_dim)
        
        # Positional Encoding
        x = self.pos_encoding(x)

        for i in range(self.encoder_stack):
            # Passing through attention layer
            [x_attn_output, attn_output_weights] = self.mha(x,x,x, key_padding_mask = key_pad_mask);

            # Add and Norm, Layer 1
            x_norm1 = self.layer_normalization(x + x_attn_output)

            # Feed forward network 
            x_ffn = self.linear2(self.relu(self.linear1(x_norm1)))

            # Add and Norm, Layer 2 
            x = self.layer_normalization(x_ffn + x_norm1)
            
        # Calculating the mean across all embeddings
        x = x.mean(dim=1)
        
        x = self.softmax_output(self.linear_output(x));

        return(x)

class ClassificationEncoder_dropout(nn.Module):

    '''
    This encodes for a classification encoder with self attention
    and dropout. 
    '''

    def __init__(self, embedding_dim, num_heads, encoder_stack = 6,
               layer_norm_epsilon = 1e-6, dropout_rate = 0.1, 
               input_vocab_size = 10000, number_of_classes = 2,
               max_positions = 5000, padding_function = None, 
               padding_value = None):

        super().__init__()

        ## Initializing static variables
        self.encoder_stack = encoder_stack; 
        self.embedding_dim = embedding_dim;
        self.padding_value = padding_value;
        self.padding_function = padding_function;

        ## Initializing Layers

        # Positional Encoding
        self.pos_encoding = PositionalEncoding(embedding_dim, max_positions)

        # Word Embedding 
        self.embedding = nn.Embedding(input_vocab_size,embedding_dim)

        # MHA Layer
        self.mha = nn.MultiheadAttention(embed_dim = embedding_dim, 
                                         num_heads = num_heads,
                                         dropout = dropout_rate);

        # Normalization Layer
        self.layer_normalization = nn.LayerNorm(embedding_dim, 
                                                eps=layer_norm_epsilon);
        # Dropout Layers
        self.dropout = nn.Dropout(dropout_rate);

        # FFN Layers with ReLU Activation
        self.linear1 = nn.Linear(embedding_dim, 4*embedding_dim);
        self.relu = nn.ReLU();
        self.linear2 = nn.Linear(4*embedding_dim, embedding_dim);

        # Linear output layer, which connects input sequence to the number
        # of classes
        self.linear_output = nn.Linear(embedding_dim,number_of_classes); 

        # Softmax output layer 
        self.softmax_output = nn.Softmax(dim=-1);

    def forward(self, x):
        '''
        Performs a single forward pass through the encoder
        '''

        # Key padding mask
        if self.padding_function is not None:
          key_pad_mask = self.padding_function(x,padding_value = self.padding_value).T
        else:
          key_pad_mask = None

        # Performs embedding
        x = self.embedding(x) * math.sqrt(self.embedding_dim)
        
        # Positional Encoding
        x = self.pos_encoding(x)
        

        for i in range(self.encoder_stack):
            # Passing through attention layer
            [x_attn_output, attn_output_weights] = self.mha(x,x,x, key_padding_mask = key_pad_mask);

            # Add and Norm, Layer 1, with dropout
            x_norm1 = self.layer_normalization(x + self.dropout(x_attn_output))

            # Feed forward network 
            x_ffn = self.linear2(self.relu(self.linear1(x_norm1)))

            # Dropout
            # Not sure when to include that
            # # x_dropout = self.dropout(x);

            # Add and Norm, Layer 2 
            x = self.layer_normalization(self.dropout(x_ffn) + x_norm1)
            
        x = x.mean(dim=1)
        
#         print('x mean shape: ',x_mean.shape)
        x = self.softmax_output(self.linear_output(x));

        return(x)