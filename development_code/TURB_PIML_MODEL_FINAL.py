# -*- coding: utf-8 -*-
"""
Created on Thu Apr 17 20:57:33 2025

@author: walter
"""
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset

from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

#%% Loading Data
# Load Training Data
script_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(script_path)
#%%
# def load_data(caseName=[], ReNum=''):       # Function to load dataset in pandas
#     wd = fr'{parent_dir}\..\..\turb_modeling_ref\\turbulence-modeling-PIML-master\\database\\pehill'      # Directory where data exist
#     d = []
#     for case in caseName:       # Loop through case names
#         if case == 'marker':    # Making sure files are read correctly
#             header = pd.read_csv(f'{wd}/{case}/{ReNum}/{case+"file"}', header=None).head(1)     # Since it will be a dataframe, fix column name
#         else:
#             header = pd.read_csv(f'{wd}/{case}/{ReNum}/{case}', header=None).head(1)            # If not, it will be loaded as a df array
#         header = header[0].str.replace(r'\s+', ' ', regex=True)     # Remove whitespace
#         header = header[0].replace('#', '')                         # Remove #
#         header = [s for s in header.split(' ') if s]                # Split into columns names
        
#         if case == 'marker':
#             df = pd.read_csv(f'{wd}/{case}/{ReNum}/{case+"file"}', sep=' ', header=None, skiprows=1, names=header)
#         else:
#             df = pd.read_csv(f'{wd}/{case}/{ReNum}/{case}', sep=' ', header=None)
#         d = d +[df]
#     return d[0], d[1]


# df_train_responses, df_train_features = load_data(caseName=['deltaField', 'marker'], ReNum='Re5600')
# df_test_responses, df_test_features = load_data(caseName=['deltaField', 'marker'], ReNum='Re10595')

# Saving to CSV 
# df_train_features[['SQUARED_DUCT_RSTRESS','PERIODIC_HILL_RSTRESS']] = df_train_responses
# df_train_features.to_csv(f'{parent_dir}/Re5600_DNS_Training_set.csv')
# df_test_features[['SQUARED_DUCT_RSTRESS','PERIODIC_HILL_RSTRESS']] = df_test_responses
# df_test_features.to_csv(f'{parent_dir}/Re10595_DNS_Test_set.csv')


df_train_features = pd.read_csv(f'{parent_dir}/Re5600_DNS_Training_set.csv')
df_test_features = pd.read_csv(f'{parent_dir}/Re10595_DNS_Test_set.csv')


# cols = df_train_features.columns[:5]
# df_train_features = df_train_features[cols]
# df_test_features = df_test_features[cols]

# df_train_responses[['SQUARED_DUCT_RSTRESS','PERIODIC_HILL_RSTRESS']] = df_train_responses
# df_train_responses = df_train_responses.drop(columns=[0,1], axis=1)

# df_test_responses[['SQUARED_DUCT_RSTRESS','PERIODIC_HILL_RSTRESS']] = df_test_responses
# df_test_responses = df_test_responses.drop(columns=[0,1], axis=1)

print(f'df_train_features DataFrame shape: {df_train_features.shape}')
print(f'df_test_features DataFrame shape: {df_test_features.shape}')


#%% Useful Functions

def norm_funct(df, cols=['QC', 'TI', 'Re_d', 'PGrad', 'TurbTime']):
    '''
    Normalize data using z-score method. Centers mean at 0 and standard deviation at 1.
    Keeps the distribution of the data the same as the original
    
    Parameters
    ----------
    df : Your data as Pandas Dataframe.
    cols : Columns you want to normalize (Train inputs or Target features).

    Returns
    -------
    df with normalized columns added
    '''
    for c in cols:
        mean = df[c].mean()
        std = df[c].std()
        df[f'{c}_NORM'] = (df[c] - mean)/std
    return df

def unst_func(arr):
    '''
    Parameters
    ----------
    arr : Output array normalized
        Input the output array if you normalized it.

    Returns
    -------
    unst_arr : Will return the ourput array unnormalized

    '''
    if arr.shape[1] == 2:
        unst_arr = np.zeros((arr.shape[0],2))
        mean1 = arr[:,0].mean()
        std1 = arr[:,0].std()
        x1 = (arr[:,0] * std1) + mean1
    
        mean2 = arr[:,1].mean()
        std2 = arr[:,1].std()
        x2 = (arr[:,1] * std2) + mean2
        
        unst_arr[:,0] = x1
        unst_arr[:,1] = x2
        
        return unst_arr
    
    else:
        unst_arr = np.zeros((arr.shape[0]))
        mean = arr.mean()
        std = arr.std()
        x = (arr * std) + mean
        unst_arr = x
        
        return unst_arr

class MyDataset(Dataset):
    '''
    Transform Training set into tensor format to read it into the PyTorch Neural Network
    '''
    def __init__(self, X, y):
        self.features = t.tensor(X, dtype=t.float32)
        self.targets = t.tensor(y, dtype=t.float32)

    def __getitem__(self, index):
        x = self.features[index]
        y = self.targets[index]
        return x, y

    def __len__(self):
        return self.targets.shape[0]
#%% Neural Network
'''
Feed-Forward or Multi-Layer Perceptron Neural Network

input_dim: Dimension of your training data.

Sequencial because it will move forward adjusting weights with linear calculations
Leaky-Relu helps learn non-linear data and assist in vanishing gradients used to adjust weights
'''
class Net(nn.Module):
    def __init__(self, input_dim):
        super(Net, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 2),
            # nn.Tanh()
        )
        
    def forward(self, x):
        return self.layers(x)

#%% Creating Training Set

cols = ['QC', 'TI', 'Re_d', 'PGrad', 'TurbTime'] # Training with these 5 columns
targ_cols = ['SQUARED_DUCT_RSTRESS','PERIODIC_HILL_RSTRESS'] # Our target column

X_train = df_train_features[cols].values
y_train = df_train_features[targ_cols].values
train_ds = MyDataset(X_train, y_train)

train_loader = DataLoader(dataset=train_ds, batch_size=200, shuffle=True)

#%%

t.manual_seed(1)
model = Net(input_dim=X_train.shape[1])

optimizer = t.optim.Adam(model.parameters(), lr=0.01)

num_epochs = 1000
percentage = 0.65
loss_list = []
train_acc_list, val_acc_list = [], []
for epoch in range(num_epochs):

    model = model.train()
    for batch_idx, (features, targets) in enumerate(train_loader):

        logits = model(features)
        loss = F.smooth_l1_loss(logits, targets)*percentage + F.mse_loss(logits, targets)*(1-percentage)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if not batch_idx % 250:
            ### LOGGING
            print(
                f"Epoch: {epoch+1:03d}/{num_epochs:03d}"
                f" | Batch {batch_idx:03d}/{len(train_loader):03d}"
                f" | Train Loss: {loss:.4f} | R2 Score: {r2_score(targets.detach().numpy().flatten(), logits.detach().numpy().flatten())}"
            )
        loss_list.append(loss.item())

#%%
X_input = df_test_features[cols]
target_data = df_test_features[['SQUARED_DUCT_RSTRESS','PERIODIC_HILL_RSTRESS']].values

input_tensor = t.tensor(X_input.values, dtype=t.float32)

model.eval()
# predict
y_mlp_tensor = model(input_tensor)
y_mlp = y_mlp_tensor.detach().numpy().astype(np.float64)

fig, ax = plt.subplots(1,2, figsize=(14,8))
for i, n in zip(range(2),['Square Duct Reynold Stress', 'Periodic Hill Reynold Stress']):
    ax[i].scatter(target_data[:,i], y_mlp[:,i], c='grey', alpha=0.4)
    xy_max = np.max([np.max(target_data[:,i]), np.max(y_mlp[:,i])])
    ax[i].plot([0, xy_max], [0, xy_max], 'k--', label='ideal')
    polyfit = np.polyfit(target_data[:,i], y_mlp[:,i], deg=1)
    reg_ys = np.poly1d(polyfit)(np.unique(target_data[:,i]))
    ax[i].plot(np.unique(target_data[:,i]), reg_ys, alpha=0.8, color='r', label='linear fit')
    ax[i].set_xlabel("Predicted")
    ax[i].set_ylabel("Target")
    ax[i].set_title(f'{n} - R2_Score: {round(r2_score(target_data[:,i], y_mlp[:,i]), 3)}')
plt.tight_layout()
plt.legend()

#%% Turb Plots

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
 
        
#%% Reynolds Triangle
dataFolderRANS = r'C:\Users\walte\Documents\GTECH_Aero\Spring_2025\Aero_Research_Spring25\turb_modeling_ref\turbulence-modeling-PIML-master\database\pehill\XiEta-RANS\Re10595\\'
symbol = 'g+'
iterateLines(dataFolderRANS, target_data, y_mlp, name='NN', symbol='m+')

