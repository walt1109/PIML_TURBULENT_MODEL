# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 18:28:17 2025

@author: walter
"""
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

#%% Loading Data
# Load Training Data
script_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(script_path)
#%%
def load_data(caseName=[], ReNum=''):       # Function to load dataset in pandas
    wd = fr'{parent_dir}\turb_modeling_ref\\turbulence-modeling-PIML-master\\database\\pehill'      # Directory where data exist
    d = []
    for case in caseName:       # Loop through case names
        if case == 'marker':    # Making sure files are read correctly
            header = pd.read_csv(f'{wd}/{case}/{ReNum}/{case+"file"}', header=None).head(1)     # Since it will be a dataframe, fix column name
        else:
            header = pd.read_csv(f'{wd}/{case}/{ReNum}/{case}', header=None).head(1)            # If not, it will be loaded as a df array
        header = header[0].str.replace(r'\s+', ' ', regex=True)     # Remove whitespace
        header = header[0].replace('#', '')                         # Remove #
        header = [s for s in header.split(' ') if s]                # Split into columns names
        
        if case == 'marker':
            df = pd.read_csv(f'{wd}/{case}/{ReNum}/{case+"file"}', sep=' ', header=None, skiprows=1, names=header)
        else:
            df = pd.read_csv(f'{wd}/{case}/{ReNum}/{case}', sep=' ', header=None)
        d = d +[df]
    return d[0], d[1]


df_train_responses, df_train_features = load_data(caseName=['deltaField', 'marker'], ReNum='Re5600')
df_test_responses, df_test_features = load_data(caseName=['deltaField', 'marker'], ReNum='Re10595')

#%% NN Model
data=[df_train_features,df_train_responses,df_test_features]
batch_size=200
epochs=100

t.manual_seed(42)
trainFeatures = t.tensor(data[0].values, dtype=t.float32)   # Load train features as tensor object
trainResponses = t.tensor(data[1].values, dtype=t.float32)  # Load train responses as tensor object
testFeatures = t.tensor(data[2].values, dtype=t.float32)    # Load test features as tensor object

train_dataset = TensorDataset(trainFeatures, trainResponses) # Takes multiple matrix as inputs (input features, labels)

train_size = int(0.8 * len(train_dataset))  # 80% for training
val_size = len(train_dataset) - train_size  # 20% for validation

# Split dataset
train_subset, val_subset = random_split(train_dataset, [train_size, val_size])

# Create DataLoaders
train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

class Net(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)  # First hidden layer
        self.fc2 = nn.Linear(64, 64)         # Second hidden layer
        self.fc3 = nn.Linear(64, 64)         # Third hidden layer
        self.fc4 = nn.Linear(64, 64)         # Fourth hidden layer
        self.fc5 = nn.Linear(64, 32)         # Fifth hidden layer
        self.fc6 = nn.Linear(32, 2)          # Output layer
        
    def forward(self, x):
        x = F.relu(self.fc1(x))  # Activation for first hidden layer
        x = F.relu(self.fc2(x))  # Activation for second hidden layer
        x = F.relu(self.fc3(x))  # Activation for third hidden layer
        x = F.relu(self.fc4(x))  # Activation for fourth hidden layer
        x = F.relu(self.fc5(x))  # Activation for fifth hidden layer
        x = t.tanh(self.fc6(x))  # Output layer with tanh activation
        return x

train_losses = []               # Store loss output
val_losses=[]

net = Net(input_dim=trainFeatures.shape[1])     # NN Variable function
criterion = nn.MSELoss(reduction='mean')         # Sum the square error instead of avg them
optimizer = optim.Adam(net.parameters(), lr=0.001)

best_loss = float('inf')
early_stop_count = 0
for epoch in range(epochs):
    net.train()  
    epoch_loss = 0
    for batch_features, batch_responses in train_loader:
        optimizer.zero_grad()
        net.zero_grad()
        output = net(batch_features)
        loss = criterion(output, batch_responses)
        loss.backward()
        optimizer.step()        # Adjusts the models parameters based on the optimizers rule (adam)
        epoch_loss += loss.item()
        print(epoch, loss)
    
    train_losses.append(epoch_loss)
    epoch_val_loss = 0  

    for val_batch_features, val_batch_responses in val_loader:
        net.eval()  
        with t.no_grad(): 
            val_testResponsesPred = net(val_batch_features)
            val_loss = criterion(val_testResponsesPred, val_batch_responses)
            epoch_val_loss += val_loss.item()
            print(epoch, val_loss)

    val_losses.append(epoch_val_loss)

    if epoch_val_loss < best_loss:
        best_loss = epoch_val_loss
        early_stop_count = 0
    else:
        early_stop_count += 1

    if early_stop_count >= 10:  # Stop training if validation loss doesn't improve for 10 epochs
        print(f"Early stopping at epoch {epoch+1}")
        break

plt.figure(figsize=(10, 6))
plt.plot(range(len(train_losses)), train_losses, label='Training Loss')
plt.plot(range(len(val_losses)), val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()
plt.grid()
