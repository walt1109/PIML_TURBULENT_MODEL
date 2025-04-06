# -*- coding: utf-8 -*-
"""
Created on Sun Apr  6 16:29:07 2025

@author: walte
"""

import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

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
def load_data(caseName=[], ReNum=''):       # Function to load dataset in pandas
    wd = fr'{parent_dir}\..\turb_modeling_ref\\turbulence-modeling-PIML-master\\database\\pehill'      # Directory where data exist
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

cols = df_train_features.columns[:5]
df_train_features = df_train_features[cols]
df_test_features = df_test_features[cols]


print(f'df_train_features DataFrame shape: {df_train_features.shape}')
print(f'df_test_features DataFrame shape: {df_test_features.shape}')

#%% NN Metrics
def predict(model, data_loader, train=True):
    target_list = []
    pred_list = []

    if train == True:

        model.eval()  
        with t.no_grad():
            for i, data_output in enumerate(data_loader):
                X, y_act = data_output
                y_act = y_act.flatten().tolist()
                y_pred = model.forward(X).flatten().tolist()
    
                targets = y_act
                predictions = y_pred
                target_list.extend(targets)
                pred_list.extend(predictions)
        model.train()
        return target_list, pred_list
    else:
        model.eval()  
        with t.no_grad():
            for i, data_output in enumerate(data_loader):
                X, y_act = data_output
                y_pred = model.forward(X)
    
                targets = y_act
                predictions = y_pred
                target_list.extend(targets.numpy())
                pred_list.extend(predictions.numpy())
        model.train()
        return np.stack(target_list), np.stack(pred_list)


def evaluate(target, pred):
    r2 = r2_score(target, pred)
    mae = mean_absolute_error(target, pred)
    rmse = mean_squared_error(target, pred)
    output = (r2, mae, rmse)
    return output


def print_scores(scores, label=''):
    r2, mae, rmse = scores
    print(f'{label} r2: {r2:0.4f}')
    print(f'{label} mae: {mae:0.4f}')
    print(f'{label} rmse: {rmse:0.4f}')
    return scores

def perc_diff(a, b):
    a = np.array(a)
    b = np.array(b)
    eps = 1e-12
    return 100 * np.abs(a - b) / ((np.abs(a) + np.abs(b)) / 2 + eps)

def plot_pred_act(act, pred, model, reg_line=True, label=''):
    xy_max = np.max([np.max(act), np.max(pred)])

    plot = plt.figure(figsize=(6,6))
    plt.plot(act, pred, 'o', ms=9, mec='k', mfc='silver', alpha=0.4)
    plt.plot([0, xy_max], [0, xy_max], 'k--', label='ideal')
    if reg_line:
        polyfit = np.polyfit(act, pred, deg=1)
        reg_ys = np.poly1d(polyfit)(np.unique(act))
        plt.plot(np.unique(act), reg_ys, alpha=0.8, label='linear fit')
    plt.axis('scaled')
    plt.xlabel(f'Actual {label}')
    plt.ylabel(f'Predicted {label}')
    plt.title(f'{type(model).__name__}, r2: {r2_score(act, pred):0.4f}')
    plt.legend(loc='upper left')
    
    return plot

def loss_curves(epochs=int, losses_train=[], losses_val=[]):
    for i, name in zip(range(3), ['R-Squared', 'MAE', 'RMSE']):
           
        perc = perc_diff(losses_train[i], losses_val[i]).mean()
        
        plt.figure(figsize=(10, 6))
        plt.plot(epochs_plt, losses_train[i], label=f'Training Loss ({name})')
        plt.plot(epochs_plt, losses_val[i], label=f'Validation Loss ({name})')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title(f'Training/Validation Loss - Percent Difference: {perc}')
        plt.legend()
        plt.grid()
#%% NN Model
# Set a random seed to ensure reproducibility across runs
RNG_SEED = 42
np.random.seed(RNG_SEED)
t.manual_seed(RNG_SEED)

data=[df_train_features,df_train_responses,df_test_features]
new_data= []
batch_size=200
epochs=500
pred=False
model=[]


t.manual_seed(42)
trainFeatures = t.tensor(data[0].values, dtype=t.float32)   # Load train features as tensor object
trainResponses = t.tensor(data[1].values, dtype=t.float32)  # Load train responses as tensor object
testFeatures = t.tensor(data[2].values, dtype=t.float32)    # Load test features as tensor object

train_dataset = TensorDataset(trainFeatures, trainResponses) # Takes multiple matrix as inputs (input features, labels)
train_size = int(0.8 * len(train_dataset))  # 80% for training
val_size = len(train_dataset) - train_size  # 20% for validation

# Split dataset
train_subset, val_subset = random_split(train_dataset, [train_size, val_size])

train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True) # Creates random batches of your dataset
val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

#%% 

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

model = Net(input_dim=trainFeatures.shape[1])
#%% Loss/Optimizer

def log_cosh_loss(pred, true):
    error = pred - true
    return t.mean(t.log(t.cosh(error)))


criterion = nn.MSELoss(reduction='mean') 
# criterion = nn.L1Loss()
# criterion = nn.SmoothL1Loss()
        
print('Loss criterion: ')
print(criterion)

# Initialize the optimzer
optim_lr = 0.01
optimizer = optim.Adam(model.parameters(), lr=optim_lr)
print('\nOptimizer: ')
print(optimizer)

#%% Training/Validation

data_type = t.float
epochs = 1000

print_every = 20
plot_every = 50

epochs_plt = []

r2_loss_train = []
r2_loss_val = []

mae_loss_train = []
mae_loss_val = []

mse_loss_train = []
mse_loss_val = []


for epoch in range(epochs):
    if epoch % print_every == 0 or epoch == epochs - 1:
        print(f'epoch: {epoch}')
    if epoch % plot_every == 0:        
        target_train, pred_train = predict(model, train_loader, train=True)
        train_scores = evaluate(target_train, pred_train)
        print_scores(train_scores, label='train')
        

        target_val, pred_val = predict(model, val_loader,train=True)
        val_scores = evaluate(target_val, pred_val)
        print_scores(val_scores, label='val')
        # plot_pred_act(target_val, pred_val, model, label='Reynolds Stess')
        
        epochs_plt.append(epoch)
        r2_loss_train.append(train_scores[0])
        mae_loss_train.append(train_scores[1])
        mse_loss_train.append(train_scores[2])
        
        r2_loss_val.append(val_scores[0])
        mae_loss_val.append(val_scores[1])
        mse_loss_val.append(val_scores[2])

    for i, data_output in enumerate(train_loader):
        X, y = data_output
        
        optimizer.zero_grad()
        output = model.forward(X).flatten()
        # loss = log_cosh_loss(output.view(-1), y.view(-1))
        loss = criterion(output.view(-1), y.view(-1))
        loss.backward()
        optimizer.step()

losses_train = [r2_loss_train, mae_loss_train, mse_loss_train]
losses_val = [r2_loss_val, mae_loss_val, mse_loss_val]
# loss_curves(epochs_plt, losses_train, losses_val)


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

#%% Testing on Test data
testResponses = t.tensor(df_test_responses.values, dtype=t.float32)
test_dataset = TensorDataset(testFeatures, testResponses)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

target_test, pred_test = predict(model, test_loader, train=False)
test_scores = evaluate(target_test[:,0], pred_test[:,0])
print_scores(test_scores, label='test')

target_test, pred_test = predict(model, test_loader, train=False)
test_scores = evaluate(target_test[:,1], pred_test[:,1])
print_scores(test_scores, label='test')

# plot = plot_pred_act(target_test[:,0], pred_test[:,0], model, label='Reynolds Stess')
# plot = plot_pred_act(target_test[:,1], pred_test[:,1], model, label='Reynolds Stess')

dataFolderRANS = r'C:\Users\walte\Documents\GTECH_Aero\Spring_2025\Aero_Research_Spring25\turb_modeling_ref\turbulence-modeling-PIML-master\database\pehill\XiEta-RANS\Re10595\\'
symbol = 'g+'
iterateLines(dataFolderRANS, df_test_responses.values, pred_test, name='NN', symbol='m+')
