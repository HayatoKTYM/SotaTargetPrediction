"""
学習ファイル
train.py batch_size = 32(64), frames = 1 で LSTM を用いずFC層のみで学習するプログラム
"""

import torch
import torch.nn as nn

import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def train(net, 
        dataloaders_dict, 
        criterion, optimizer,
        num_epochs=10,
        output='./',
        resume=False, 
        ):
    """
    学習ループ
    """
    #f = open('log.txt','w')
    os.makedirs(output,exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('using',device)
    net.to(device)

    Loss = {'train': [0]*num_epochs,
            'val': [0]*num_epochs}

    Acc = {'train': [0]*num_epochs,
            'val': [0]*num_epochs}

    criterionA = nn.CrossEntropyLoss()
    criterionB = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-------------')
        # epochごとの訓練と検証のループ
        for phase in ['train', 'val']:
            if phase == 'train':
                net.train()  # モデルを訓練モードに
            else:
                net.eval()   # モデルを検証モードに

            epoch_loss = 0.0  # epochの損失和
            epoch_corrects = 0  # epochの正解数
            epoch_lossA = 0.0  # epochの損失和
            epoch_correctsA = 0  # epochの正解数
            epoch_lossB = 0.0  # epochの損失和
            epoch_correctsB = 0  # epochの正解数
            
            for inputs, labels in dataloaders_dict[phase]:    
                inputs = inputs.to(device,dtype=torch.float32)
                labels = labels.to(device,dtype=torch.long)
                
                outA,outB,out = net(inputs) # 順伝播
                loss_ = criterion(out, labels[:,0]) 
                lossA = criterionA(outA, labels[:,1])
                lossB = criterionA(outB, labels[:,2])
                loss = loss_ + lossA + lossB
                _, preds = torch.max(out, 1)  # ラベルを予測
                _, predsA = torch.max(outA, 1)  # ラベルを予測
                _, predsB = torch.max(outB, 1)  # ラベルを予測
                
                if phase == 'train' : # 訓練時はバックプロパゲーション
                    optimizer.zero_grad() # 勾配の初期化
                    loss.backward() # 勾配の計算
                    optimizer.step()# パラメータの更新

                epoch_loss += loss_.item() * inputs.size(0)  # lossの合計を更新
                epoch_corrects += torch.sum(preds == labels[:,0].data) # 正解数の合計を更新
                epoch_lossA += lossA.item() * inputs.size(0)  # lossの合計を更新
                epoch_correctsA += torch.sum(predsA == labels[:,1].data) # 正解数の合計を更新
                epoch_lossB += lossB.item() * inputs.size(0)  # lossの合計を更新
                epoch_correctsB += torch.sum(predsB == labels[:,2].data) # 正解数の合計を更新

                
            # epochごとのlossと正解率を表示
            epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
            epoch_acc = epoch_corrects.double() / len(dataloaders_dict[phase].dataset)
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            epoch_lossA = epoch_lossA / len(dataloaders_dict[phase].dataset)
            epoch_accA = epoch_correctsA.double() / len(dataloaders_dict[phase].dataset)
            print('{} VAD_A_Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_lossA, epoch_accA))
            epoch_lossB = epoch_lossB / len(dataloaders_dict[phase].dataset)
            epoch_accB = epoch_correctsB.double() / len(dataloaders_dict[phase].dataset)
            print('{} VAD_B_Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_lossB, epoch_accB))

            Loss[phase][epoch] = epoch_loss 
            Acc[phase][epoch] = float(epoch_acc.cpu().numpy())
            
        if resume:
            torch.save(net.state_dict(), os.path.join(output,'epoch_{}_acc{}_loss{}_ut_train.pth'.format(epoch+1,epoch_acc,epoch_loss)))
        
            y_true, y_prob = np.array([]), np.array([])
            for inputs, labels in dataloaders_dict['test']:
                inputs = inputs.to(device,dtype=torch.float32)
                outA,outB,out = net(inputs) # 順伝播
                _, preds = torch.max(out, 1)  # ラベルを予測
                
                y_true = np.append(y_true, labels[:,0].data.numpy())
                y_prob = np.append(y_prob, nn.functional.softmax(out,dim=-1).cpu().data.numpy()[:,1])

            #保存
            plt.figure(figsize=(20,4))
            plt.rcParams["font.size"] = 18
            plt.plot(y_true[:1000],label = 'true label',color='r',linewidth=3.0)
            plt.plot(y_prob[:1000],label = 'predict',color='m')
            plt.fill_between(list(range(1000)),y_prob[:1000],color='m',alpha=0.35)
            plt.legend()
            plt.savefig(os.path.join(output,'result_{}_acc{:.3f}_loss{:.3f}_ut_train.png'.format(epoch+1,epoch_acc,epoch_loss)))

        #print(confusion_matrix(y_true, y_pred))
        #print(classification_report(y_true, y_pred))
        print('-------------')
            
    if resume: # 学習過程(Loss) を保存するか    
        plt.figure(figsize=(15,4))
        plt.plot(Loss['val'],label='val')
        plt.plot(Loss['train'],label='train')
        plt.legend()
        plt.savefig(os.path.join(output,'history.png'))

    print('training finish and save train history...')
    y_true, y_pred = np.array([]), np.array([])
    y_prob = np.array([])

    for inputs, labels,labelsA,labelsB in dataloaders_dict['test']:
        inputs = inputs.to(device,dtype=torch.float32)
        outA, outB, out = net(inputs)
        _, preds = torch.max(out, 1)  # ラベルを予測

        y_true = np.append(y_true, labels.data.numpy())
        y_pred = np.append(y_pred, preds.cpu().data.numpy())
        y_prob = np.append(y_prob, nn.functional.softmax(out).cpu().data.numpy()[:,1])

    print(accuracy_score(y_true, y_pred))
    print(confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred))

def train_lstm(net, 
        dataloaders_dict, 
        criterion, optimizer,
        num_epochs=10,
        output='./',
        resume=True,         
    ):
    """
    学習ループ
    """
    os.makedirs(output,exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('using',device)
    net.to(device)

    Loss = {'train': [0]*num_epochs,
            'val': [0]*num_epochs}

    Acc = {'train': [0]*num_epochs,
            'val': [0]*num_epochs}

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-------------')
        # epochごとの訓練と検証のループ
        for phase in ['train', 'val']:
            if phase == 'train':
                net.train()  # モデルを訓練モードに
            else:
                net.eval()   # モデルを検証モードに

            epoch_loss = 0.0  # epochの損失和
            epoch_corrects = 0  # epochの正解数
            hidden = None # lstmの初期状態
            loss = 0 # 損失和
            train_cnt = 0 # feed farward 回数

            for inputs, labels in dataloaders_dict[phase]:    
                inputs = inputs.to(device,dtype=torch.float32)
                labels = labels.to(device,dtype=torch.long)
                if labels > 1: #会話データの境目に -1 を 挿入:
                    hidden = None
                    continue

                if hidden is None:
                    out,hidden= net(inputs,None) # 順伝播
                else:
                    out,hidden = net(inputs,hidden) # 順伝播
                train_cnt += 1
                l = criterion(out, labels) # ロスの計算
                loss += l
                _, preds = torch.max(out, 1)  # ラベルを予測
                
                if phase == 'train' and train_cnt % 32 == 0 : # 訓練時はバックプロパゲーション
                    optimizer.zero_grad() # 勾配の初期化
                    loss.backward() # 勾配の計算
                    optimizer.step()# パラメータの更新
                    loss = 0 #累積誤差reset
                    
                    hidden = (hidden[0].detach(),hidden[1].detach()) # BPのつながりをカット

                epoch_loss += l.item() * inputs.size(0)  # lossの合計を更新
                epoch_corrects += torch.sum(preds == labels.data) # 正解数の合計を更新
                
            # epochごとのlossと正解率を表示
            epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
            epoch_acc = epoch_corrects.double() / len(dataloaders_dict[phase].dataset)
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            Loss[phase][epoch] = epoch_loss 
            Acc[phase][epoch] = float(epoch_acc.cpu().numpy())
            
        if resume:
            torch.save(net.state_dict(), os.path.join(output,'epoch_{}_acc{}_loss{}_ut_train.pth'.format(epoch+1,epoch_acc,epoch_loss)))
        
            y_true, y_prob = np.array([]), np.array([])
            # precision , recall , F1-score, confusion_matrix を表示
            hidden = None
            for inputs, labels in dataloaders_dict['test']:
                inputs = inputs.to(device,dtype=torch.float32)
                if labels > 1: #会話データの境目に -1 を 挿入:
                    hidden = None
                    continue
                if hidden is None:
                    out,hidden = net(inputs)
                else:
                    out,hidden = net(inputs,hidden)
                _, preds = torch.max(out, 1)  # ラベルを予測
                y_true = np.append(y_true, labels.data.numpy())
                y_prob = np.append(y_prob, nn.functional.softmax(out,dim=-1).cpu().data.numpy()[0][1])
            
            # u_a_(t) と u_b_(t) から　u(t) を算出
            y_a = y_true[:len(y_true)//2]
            y_b = y_true[len(y_true)//2:]
            y_true = [min([y_a[i],y_b[i]]) for i in range(len(y_a))]
            y_true = np.clip(y_true,0,1)
            
            y_a = y_prob[:len(y_prob)//2]
            y_b = y_prob[len(y_prob)//2:]
            y_prob = [min([y_a[i],y_b[i]]) for i in range(len(y_a))]
            
            plt.figure(figsize=(20,4))
            plt.plot(y_true[:300],label = 'true label')
            plt.plot(y_prob[:300],label = 'predict')
            plt.legend()
            plt.savefig(os.path.join(output,'result_{}_acc{}_loss{}_ut_train.png'.format(epoch+1,epoch_acc,epoch_loss)))

        print('-------------')

    if resume: # 学習過程(Loss) を保存するか    
        plt.figure(figsize=(15,4))
        plt.plot(Loss['val'],label='val')
        plt.plot(Loss['train'],label='train')
        plt.legend()
        plt.savefig(os.path.join(output,'history.png'))
    
    print('training finish and save train history...')
    y_true, y_pred = np.array([]), np.array([])
    y_prob = np.array([])
    
    hidden = None
    for inputs, labels in dataloaders_dict['test']: # 評価
        inputs = inputs.to(device,dtype=torch.float32)
        if labels > 1: #会話データの境目に -1 を 挿入:
            hidden = None
            continue
        if hidden is None:
            out,hidden = net(inputs)
        else:
            out,hidden = net(inputs,hidden)
        _, preds = torch.max(out, 1)  # ラベルを予測

        y_true = np.append(y_true, labels.data.numpy())
        y_pred = np.append(y_pred, preds.cpu().data.numpy())
        y_prob = np.append(y_prob, nn.functional.softmax(out,dim=-1).cpu().data.numpy()[0][1])

    print(accuracy_score(y_true, y_pred))
    print(confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred))
    
    # u_a_(t) と u_b_(t) から　u(t) を算出
    y_a = y_true[:len(y_true)//2]
    y_b = y_true[len(y_true)//2:]
    y_true = [min([y_a[i],y_b[i]]) for i in range(len(y_a))]
    y_true = np.clip(y_true,0,1)
    
    y_a = y_prob[:len(y_prob)//2]
    y_b = y_prob[len(y_prob)//2:]
    y_prob = [min([y_a[i],y_b[i]]) for i in range(len(y_a))]
    
    if resume: # 出力の可視化例を保存するかどうか
        plt.figure(figsize=(15,4))
        plt.plot(y_true[:300],label = 'true label')
        plt.plot(y_prob[:300],label = 'predict')
        plt.legend()
        plt.savefig(os.path.join(output,'result.png'))
