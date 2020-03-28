from model import U_t_train, TimeActionPredict
from train import train, train_lstm
from utils import setup

import torch
from torch.utils import data
import torch.nn as nn
import torch.optim as optim

import pandas as pd
import numpy as np

from torch.utils import data

def hang_over(y):
    """
    u の末端 400 ms を １ にする
    """
    print('before',y.sum())
    for i in range(len(y)-1):
        if y[i] == 0 and y[i+1] == 1:
            y[i-3:i+1] = 1.
    print('after',y.sum())
    return y

def main():
    dense_flag = True
    df_list,lld_list = setup()
    train_id = 89
    df = pd.concat(df_list[:train_id])
    lld = pd.concat(lld_list[:train_id])
    x = lld.iloc[:,:114].values
    x = np.append(x,lld.iloc[:,114:].values,axis=0)
    x = x.reshape(-1,10,114)    
    print(x.shape)
    y = hang_over(1.0 - df['utter_A'].values)
    y = np.append(y,hang_over(1.0 - df['utter_B'].values))
    print(y.shape)
    print(set(y))

    df = pd.concat(df_list[train_id:])
    lld = pd.concat(lld_list[train_id:])
    x_val = lld.iloc[:,:114].values
    x_val = np.append(x_val,lld.iloc[:,114:].values,axis=0)
    x_val = x_val.reshape(-1,10,114)
    
    y_val = hang_over(1.0 - df['utter_A'].values)
    y_val = np.append(y_val,hang_over(1.0 - df['utter_B'].values))
    
    if dense_flag:
        net = U_t_train(input_size=x.shape[-1])
    else:
        net = TimeActionPredict()                
    print('Model :', net.__class__.__name__)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    for name, param in net.named_parameters():
        if 'fc' in name or 'lstm' in name:
            param.requires_grad = True
            print("勾配計算あり。学習する：", name)
        else:
            param.requires_grad = False
        
    train_dataloader = data.DataLoader(
        list(zip(x,y)), batch_size=32, shuffle=dense_flag)

    test_dataloader = data.DataLoader(
        list(zip(x_val,y_val)), batch_size=32, shuffle=False)

    dataloaders_dict = {"train": train_dataloader, "val": test_dataloader, "test": test_dataloader}
    if not dense_flag:
        train_lstm(net=net, dataloaders_dict=dataloaders_dict, criterion=criterion,optimizer=optimizer,
            num_epochs=100, output='./lstm_model_hang_over/', resume=True)
    else:
        train(net=net, dataloaders_dict=dataloaders_dict, criterion=criterion,optimizer=optimizer,
            num_epochs=100, output='./dense_model/', resume=True)
if __name__ == '__main__':
    main()
