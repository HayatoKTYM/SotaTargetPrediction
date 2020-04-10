from model import U_t_train, TimeActionPredict
from train import train
from utils import setup

import torch
from torch.utils import data
import torch.nn as nn
import torch.optim as optim
import os
import pandas as pd
import numpy as np
import argparse
from torch.utils import data

def hang_over(y):
    """
    u の末端 400 ms を １ にする
    """
    #print('before',y.sum())
    for i in range(len(y)-1):
        if y[i] == 0 and y[i+1] == 1:
            y[i-3:i+1] = 1.
    #print('after',y.sum())
    return y

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='/mnt/aoni04/katayama/DATA2020/')
    parser.add_argument('-o', '--out', type=str, default='./lstm_model')
    parser.add_argument('-w', '--u_PATH', type=str,
     default='../u_t/SPEC/../u_t/SPEC/202004021729/epoch_21_acc0.909_loss0.218_ut_train.pth')
    parser.add_argument('-e', '--epoch', type=int, default=100)
    parser.add_argument('-r', '--resume', type=str, default=True)
    parser.add_argument('--hang', type=str, default=False)

    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    import datetime
    now = datetime.datetime.now()
    print('{0:%Y%m%d%H%M}'.format(now))
    out = os.path.join(args.out, '{0:%Y%m%d%H%M}'.format(now))
    os.makedirs(out, exist_ok=True)

    dense_flag = False
    df_list, lld_list = setup(PATH=args.input, dense_flag=False)
    train_id = 100
    # 連結せずに 会話毎に list でもつ
    df_train = df_list[:train_id]
    feature = []
    df_val = df_list[train_id:]
    feature_val = []
    lld_train = lld_list[:train_id]
    lld_val = lld_list[train_id:]
    lld_dict = {'train': lld_train, 'val': lld_val}
    df_dict = {'train': df_train, 'val': df_val}
    dataloaders_dict = {"train": feature, "val": feature_val}

    for phase in df_dict.keys():
        df = df_dict[phase]
        lld = lld_dict[phase]
        feature = dataloaders_dict[phase]

        for i in range(len(df)):
            x = df[i].iloc[:, -512:].values
            x = lld[i].values.reshape(-1,10,228)
            img = df[i].iloc[:, -64:].values

            y = df[i]['target'].map(lambda x:0 if x == 'A' else 1)
            target = np.r_[[y[0]], y[:-1]] #１時刻前の顔向き
            target = target.reshape(len(target), 1)
            yA = hang_over(1.0 - df[i]['utter_A'].values)
            yB = hang_over(1.0 - df[i]['utter_B'].values)
            y = np.c_[y, yA, yB, df[i]['gaze'].values]
            #print(img.shape, target.shape)
            img = np.append(img, target, axis=1)
            print(x.shape,img.shape)
            feature.append((x, img, y))

    if dense_flag:
        net = U_t_train(input_size=114)
    else:
        net = TimeActionPredict(input_size=114, hidden_size=64)                
    print('Model :', net.__class__.__name__)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    for name, param in net.named_parameters():
        if 'fc' in name or 'lstm' in name:
            param.requires_grad = True
            print("勾配計算あり。学習する：", name)
        else:
            param.requires_grad = False

    print('train data is ', np.shape(dataloaders_dict['train']))
    print('test data is ', np.shape(dataloaders_dict['val'])) 
    train(
        net=net, 
        dataloaders_dict=dataloaders_dict, 
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=args.epoch, 
        output=out, 
        resume=args.resume
    )


if __name__ == '__main__':
    main()
