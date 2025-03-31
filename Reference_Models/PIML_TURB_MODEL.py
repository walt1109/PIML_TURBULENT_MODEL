# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 22:19:08 2025

@author: walte
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 18:05:57 2025

@author: walte
"""

import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

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

def Torch_NN(data=[df_train_features,df_train_responses,df_test_features], new_data= [], batch_size=200, epochs=500, pred=False, model=[]):
    t.manual_seed(42)
    trainFeatures = t.tensor(data[0].values, dtype=t.float32)   # Load train features as tensor object
    trainResponses = t.tensor(data[1].values, dtype=t.float32)  # Load train responses as tensor object
    testFeatures = t.tensor(data[2].values, dtype=t.float32)    # Load test features as tensor object
    
    train_dataset = TensorDataset(trainFeatures, trainResponses) # Takes multiple matrix as inputs (input features, labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) # Creates random batches of your dataset
    '''    
        An activation function in a neural network (NN) introduces non-linearity to the model, 
        allowing it to learn complex patterns beyond simple linear relationships  
        
        Criterion measures how well the neural network's predictions match the true labels. 
        It calculates an error (or "loss")
        
        Optimizer updates the model's parameters (weights & biases) using gradient descent
        adam: tracks mean and variance of the gradients 
        Gradient represents the direction and rate of change of the function (loss gradient)
        
        In PyTorch, gradients accumulate by default instead of being reset after each backward pass. 
        This means if we don’t clear them, gradients from multiple batches will add up, 
        leading to incorrect updates. Unintentional larger updates and Incorrect 
        weight updates, causing training instability
    ''' 
    if pred == True:
        # Here we use predict
        net = model[0]
        net.eval()  
        with t.no_grad(): 
            if len(new_data)>0:
                new_df = t.tensor(new_data[0].values, dtype=t.float32)
                testResponsesPred = net(new_df)     
                return testResponsesPred.numpy()

            else:
                testResponsesPred = net(testFeatures)        
                return testResponsesPred.numpy()
        
    class Net(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, 64)  # First hidden layer
            self.fc2 = nn.Linear(64, 32)         # Second hidden layer
            self.fc3 = nn.Linear(32, 2)          # Output layer
            
        def forward(self, x):
            x = F.relu(self.fc1(x))  # Activation for first hidden layer
            x = F.relu(self.fc2(x))  # Activation for second hidden layer
            x = t.tanh(self.fc3(x))  # Output layer with tanh activation
            return x
    
    train_losses = []               # Store loss output
    
    net = Net(input_dim=trainFeatures.shape[1])     # NN Variable function
    criterion = nn.MSELoss(reduction='mean')         # Sum the square error instead of avg them
    optimizer = optim.Adam(net.parameters(), lr=0.004)

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
            print(loss)
        train_losses.append(loss.item())

            
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            early_stop_count = 0
        else:
            early_stop_count += 1

        if early_stop_count >= 50:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(train_losses)), train_losses, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.grid()
    
    return net

    


#%%

def plotXiEta(XiEta_RANS, testResponses, testResponsesPred, name, symbol='r^'):
    # Reconstruct Barycentric coordinates
    XiEta_DNS = XiEta_RANS + testResponses
    XiEta_ML = XiEta_RANS + testResponsesPred
    # Plot Reynolds stress anisotropy in Barycentric triangle
    interval = 2
    pointsNum = int(XiEta_RANS.shape[0])
    plt.figure()
    plt.plot([0,1,0.5,0.5,0],[0,0,3**0.5/2.0,3**0.5/2.0,0],'g-')
    p1, = plt.plot(XiEta_RANS[:pointsNum:interval,0],XiEta_RANS[:pointsNum:interval,1],
                   'bo', markerfacecolor='none', markeredgecolor='b',
                   markeredgewidth=2, markersize=10)
    p2, = plt.plot(XiEta_DNS[:pointsNum:interval,0],XiEta_DNS[:pointsNum:interval,1],
                   'ks', markerfacecolor='none', markeredgecolor='k',
                   markeredgewidth=2, markersize=10)
    p3, = plt.plot(XiEta_ML[:pointsNum:interval,0],XiEta_ML[:pointsNum:interval,1],
                   symbol, markerfacecolor='none', #markeredgecolor='r',
                   markeredgewidth=2, markersize=10)
    lg = plt.legend([p1,p2,p3], ['RANS', 'DNS', name], loc = 0)
    lg.draw_frame(False)
    plt.ylim([0,3**0.5/2.0])
    plt.show()

def comparePlotRFNN(XiEta_RANS, testResponses, testResponsesPred_RF, testResponsesPred_NN):
    
    XiEta_DNS = XiEta_RANS + testResponses
    XiEta_RF = XiEta_RANS + testResponsesPred_RF
    XiEta_NN = XiEta_RANS + testResponsesPred_NN
    # Plot Reynolds stress anisotropy in Barycentric triangle
    interval = 2
    pointsNum = int(XiEta_RANS.shape[0])
    plt.figure()
    plt.plot([0,1,0.5,0.5,0],[0,0,3**0.5/2.0,3**0.5/2.0,0],'g-')
    p1, = plt.plot(XiEta_RANS[:pointsNum:interval,0],XiEta_RANS[:pointsNum:interval,1],
                   'bo', markerfacecolor='none', markeredgecolor='b',
                   markeredgewidth=1.5, markersize=8)
    p2, = plt.plot(XiEta_DNS[:pointsNum:interval,0],XiEta_DNS[:pointsNum:interval,1],
                   'ks', markerfacecolor='none', markeredgecolor='k',
                   markeredgewidth=1.5, markersize=8)
    p3, = plt.plot(XiEta_RF[:pointsNum:interval,0],XiEta_RF[:pointsNum:interval,1],
                   'r^', markerfacecolor='none', markeredgecolor='r',
                   markeredgewidth=1.5, markersize=8)
    p4, = plt.plot(XiEta_NN[:pointsNum:interval,0],XiEta_NN[:pointsNum:interval,1],
                   'r+', markerfacecolor='none', markeredgecolor='g',
                   markeredgewidth=1.5, markersize=8)
    lg = plt.legend([p1,p2,p3, p4], ['RANS', 'DNS', 'RF', 'NN'], loc = 0)
    lg.draw_frame(False)
    plt.ylim([0,3**0.5/2.0])
    plt.show()

def iterateLines(dataFolderRANS, testResponses, testResponsesPred, name, symbol='r^'):
    # Start index of different sample lines
    indexList = [0, 98, 191, 287, 385, 483, 581, 679, 777, 875, 971]
    # Make plots at x=2 and x=4
    for iterN in [3,5]:
        XiEta = np.loadtxt(dataFolderRANS + 'line' + str(iterN) + '_XiEta.xy')
        startIndex = indexList[iterN-1]
        endIndex = indexList[iterN]
        plotXiEta(XiEta, testResponses[startIndex:endIndex,:], 
                         testResponsesPred[startIndex:endIndex,:], name, symbol)
    #plt.show()


def compareResults(dataFolderRANS, testResponses, testResponsesPred_RF, testResponsesPred_NN):
    ## compare the results in one plot
    # Start index of different sample lines
    indexList = [0, 98, 191, 287, 385, 483, 581, 679, 777, 875, 971]
    # Make plots at x=2 and x=4
    for iterN in [3,5]:
        XiEta = np.loadtxt(dataFolderRANS + 'line' + str(iterN) + '_XiEta.xy')
        startIndex = indexList[iterN-1]
        endIndex = indexList[iterN]
        comparePlotRFNN(XiEta, testResponses[startIndex:endIndex,:], 
                    testResponsesPred_RF[startIndex:endIndex,:], 
                    testResponsesPred_NN[startIndex:endIndex,:])


#%%
net = Torch_NN(batch_size=200, epochs=500)

testResponsesPred = Torch_NN(batch_size=200, pred=True, epochs=100, model=[net])


dataFolderRANS = r'C:\Users\walte\Documents\GTECH_Aero\Spring_2025\Aero_Research_Spring25\turb_modeling_ref\turbulence-modeling-PIML-master\database\pehill\XiEta-RANS\Re10595\\'
symbol = 'g+'
iterateLines(dataFolderRANS, df_test_responses.values, testResponsesPred, name='NN', symbol='m+')
plt.show()
