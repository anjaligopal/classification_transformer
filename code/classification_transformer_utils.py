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
from torchmetrics import Accuracy

def accuracy_score(true_labels,predicted_labels):
  '''
  Calculates the accuracy score for binary classification.
  '''

  # Calculating accuracy
  accuracy = Accuracy()

  score = accuracy(torch.tensor(true_labels),torch.tensor(predicted_labels));

  score = round(float(score),3)

  return score

def train_and_evaluate(model, training_data, test_data, optimizer,
                       criterion, device = 'cpu', scheduler = None, epochs = 30, 
                       verbose = True, record_lr = False):
  '''
  Trains a model and outputs information about its training results.

  Inputs:
  - model 
  - training_data: should be a data loader
  - test_data:  should be a data loader
  - optimizer for model
  - loss criterion
  - device: cpu by default
  - scheduler: set to None if no scheduling occuring
  - epochs: default 30
  - verbose = True prints notes about training
  - record_lr prints information about learning rates for logging

  Outputs:
  - model
  - epoch_training_loss_array: training loss at every epoch
  - epoch_test_loss_array: test loss at every epoch
  - epoch_training_accuracy_array: training accuracy at every epoch
  - epoch_test_accuracy_array: test accuracy at every epoch
  - time_for_epochs: time it takes to complete every epoch
  - optimizer_lr, scheduler_lr: learning rates for every step (not epoch)
                              if record_lr = true
  '''

  import time

  # Parameters for logging
  epoch_training_loss_array = [];
  epoch_test_loss_array = [];
  epoch_training_accuracy_array = [];
  epoch_test_accuracy_array = [];
  time_for_epochs = []

  # Logging optimizer and scheduler learning rates
  optimizer_lr = []
  scheduler_lr = [] # For logging scheduler learning rates

  model = model.to(device)

  if verbose == True:
    print("Using device: ",device)

  start_time = time.time()


  for epoch in range(epochs):

    epoch_training_loss = 0;
    epoch_test_loss = 0; 
    epoch_training_labels = [];
    epoch_training_predicted = [];
    epoch_test_labels = [];
    epoch_test_predicted = []

    if verbose == True:
      print("Epoch {}/{} ".format(epoch+1,epochs))

    ## Training 
    for i, (batch_labels, batch_lines) in enumerate(training_data):
      model.train()

      # Transfer labels to GPU
      batch_labels = batch_labels.to(device)

      # Log 
      epoch_training_labels.extend(batch_labels)

      # Forward pass
      model_output = model.forward(batch_lines.to(device))
      
      # Predicted labels
      _, predicted_labels = torch.max(model_output,1)
      epoch_training_predicted.extend(predicted_labels)
      
      # Calculating Loss
      loss = criterion(model_output,batch_labels)
      epoch_training_loss += round(float(loss),3); 
              
      # Backward pass and optimize
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

      if scheduler is not None:
          scheduler.step()

          if record_lr == True:
            optimizer_lr.append(get_lr(optimizer))
            scheduler_lr.append(scheduler.get_lr())

    ## Testing
    for i, (batch_labels, batch_lines) in enumerate(test_data):
      model.eval()

      # Transfer labels to GPU
      batch_labels = batch_labels.to(device)

      # Log 
      epoch_test_labels.extend(batch_labels)

      # Forward pass
      model_output = model.forward(batch_lines.to(device))
      
      # Predicted labels
      _, predicted_labels = torch.max(model_output,1)
      epoch_test_predicted.extend(predicted_labels)

      # Calculating Loss
      loss = criterion(model_output,batch_labels)
      epoch_test_loss += round(float(loss),3); 



    ## Accuracy
    epoch_training_accuracy = accuracy_score(epoch_training_labels,epoch_training_predicted);
    epoch_test_accuracy = accuracy_score(epoch_test_labels,epoch_test_predicted);
    epoch_time = round(time.time()-start_time,3)

    ## Logging
    epoch_training_loss_array.append(epoch_training_loss) 
    epoch_test_loss_array.append(epoch_test_loss) 
    epoch_training_accuracy_array.append(epoch_training_accuracy)
    epoch_test_accuracy_array.append(epoch_test_accuracy)
    time_for_epochs.append(epoch_time)
  
    if verbose == True:
      print("Epoch training loss: ",round(epoch_training_loss,3))
      print("Epoch test loss: ",round(epoch_test_loss,3))
      print("Epoch training accuracy: ",epoch_training_accuracy)
      print("Epoch test set accuracy: ",epoch_test_accuracy)
      print("Time: ",epoch_time)

  return(model, epoch_training_loss_array, epoch_test_loss_array, epoch_training_accuracy_array, 
         epoch_test_accuracy_array, time_for_epochs, optimizer_lr, scheduler_lr)





def save_outputs(model, epoch_training_loss_array, epoch_test_loss_array, epoch_training_accuracy_array, epoch_test_accuracy_array, time_for_epochs, optimizer_lr = None, scheduler_lr = None, model_filename = None, data_filename = None):

  '''
  Saves the outputs of train_and_evaluate

  Inputs:
  - model: the neural net model
  - epoch_training_loss_array, epoch_training_accuracy_array, epoch_test_accuracy_array, time_for_epochs: arrays output from train_and_evaluate, should all be the same dim
  - optimizer_lr, scheduler_lr: for each timestep; leave None to produce no outputs
  - model_filename: name to save model 
  - data_filename: name to same data file 


  '''

  from datetime import datetime
  import numpy as np 
  import pandas as pd


  datetime_suffix = datetime.now().strftime("%Y%m%d_%H%M%S")

  if model_filename == None:
      model_filename = 'model'+datetime_suffix+'.ckpt';

  if data_filename == None:
      data_filename = 'data_'+datetime_suffix+'.csv';



  # Saving the model

  if model_filename.split('.')[-1] != 'ckpt':
      model_filename = model_filename + '.ckpt'

  torch.save(model.state_dict(), model_filename)

  # Saving the accuracy and times

  accuracy_df = pd.DataFrame(np.array([np.arange(len(epoch_training_loss_array)),epoch_training_loss_array,epoch_test_loss_array,epoch_training_accuracy_array,epoch_test_accuracy_array,time_for_epochs]).T, columns=['Epochs','Training Loss','Test Loss','Training Accuracy','Test Accuracy','Time']);

  if len(data_filename.split('.'))<2:
      data_filename = data_filename + '.csv'

  accuracy_df.to_csv(data_filename,index=False)


  # saving the optimizers

  lr_df = pd.DataFrame()

  if optimizer_lr is not None:
      lr_df['Optimizer LR'] = optimizer_lr

  if scheduler_lr is not None:
      lr_df['Scheduler LR'] = scheduler_lr

  if scheduler_lr or optimizer_lr is not None:
      if data_filename == None:   
          lr_filename = 'lr_'+datetime_suffix+'.csv';
      else:
          lr_filename = 'lr_'+data_filename;

      lr_df.to_csv(lr_filename,index=False)