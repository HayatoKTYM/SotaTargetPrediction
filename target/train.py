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
            epoch_lossC = 0.0  # epochの損失和
            epoch_correctsC = 0  # epochの正解数
            total = 0
            feature = np.array(dataloaders_dict[phase])

            if phase == 'train':
                N = np.random.permutation(len(feature))
                print(N)
            else:
                N = np.arange(len(feature))

            for f in feature[N]:
                total += len(f[0])
                for i in range(len(f[0])):
                    inputs = torch.tensor(f[0][i]).to(device, dtype=torch.float32)
                    img = torch.tensor(f[1][i]).to(device, dtype=torch.float32)
                    labels = torch.tensor(f[2][i]).to(device, dtype=torch.long)
                    labels = labels.unsqueeze(0)
                    outA, outB, outC, out = net(inputs, img) # 順伝播
                    loss_ = criterion(out, labels[:,0]) 
                    lossA = criterionA(outA, labels[:,1])
                    lossB = criterionA(outB, labels[:,2])
                    lossC = criterionA(outC, labels[:,3])
                    loss = loss_ + lossA + lossB + lossC
                    _, preds = torch.max(out, 1)  # ラベルを予測
                    _, predsA = torch.max(outA, 1)  # ラベルを予測
                    _, predsB = torch.max(outB, 1)  # ラベルを予測
                    _, predsC = torch.max(outC, 1)  # ラベルを予測
                    
                    if phase == 'train' : # 訓練時はバックプロパゲーション
                        optimizer.zero_grad() # 勾配の初期化
                        loss.backward() # 勾配の計算
                        optimizer.step()# パラメータの更新

                    epoch_loss += loss_.item()
                    epoch_corrects += torch.sum(preds == labels[:,0].data) # 正解数の合計を更新
                    epoch_lossA += lossA.item()
                    epoch_correctsA += torch.sum(predsA == labels[:,1].data) # 正解数の合計を更新
                    epoch_lossB += lossB.item()
                    epoch_correctsB += torch.sum(predsB == labels[:,2].data) # 正解数の合計を更新
                    epoch_lossC += lossC.item()
                    epoch_correctsC += torch.sum(predsC == labels[:,3].data) # 正解数の合計を更新

                
            # epochごとのlossと正解率を表示
            epoch_loss = epoch_loss / total
            epoch_acc = epoch_corrects.double() / total
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            epoch_lossA = epoch_lossA / total
            epoch_accA = epoch_correctsA.double() / total
            print('{} VAD_A_Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_lossA, epoch_accA))
            epoch_lossB = epoch_lossB / total
            epoch_accB = epoch_correctsB.double() / total
            print('{} VAD_B_Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_lossB, epoch_accB))
            epoch_lossC = epoch_lossC / total
            epoch_accC = epoch_correctsC.double() / total
            print('{} gaze_Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_lossC, epoch_accC))

            Loss[phase][epoch] = epoch_loss 
            Acc[phase][epoch] = float(epoch_acc.cpu().numpy())
            
        if resume:
            torch.save(net.state_dict(), os.path.join(output,'epoch_{}_acc{}_loss{}_ut_train.pth'.format(epoch+1,epoch_acc,epoch_loss)))
        
            y_true, y_prob = np.array([]), np.array([])
            feature = np.array(dataloaders_dict['val'])
            for f in feature:
                for i in range(len(f[0])):
                    inputs = torch.tensor(f[0][i]).to(device, dtype=torch.float32)
                    img = torch.tensor(f[1][i]).to(device, dtype=torch.float32)
                    if i > 0:
                        img[-1].data = preds.data
                    labels = torch.tensor(f[2][i]).to(device, dtype=torch.long)
                    labels = labels.unsqueeze(0)
                    outA, outB, outC, out = net(inputs, img) # 順伝播
                    _, preds = torch.max(out, 1)  # ラベルを予測
                    #print('preds',preds)
                    y_true = np.append(y_true, labels[:,0].cpu().data.numpy())
                    y_prob = np.append(y_prob, nn.functional.softmax(out,dim=-1).cpu().data.numpy()[:,1])

            #保存
            plt.figure(figsize=(20,4))
            plt.rcParams["font.size"] = 18
            plt.plot(y_true[:1000],label = 'true label',color='r',linewidth=3.0)
            plt.plot(y_prob[:1000],label = 'predict',color='m')
            plt.fill_between(list(range(1000)),y_prob[:1000],color='m',alpha=0.35)
            plt.legend()
            plt.savefig(os.path.join(output,'result_{}_acc{:.3f}_loss{:.3f}_ut_train.png'.format(epoch+1,epoch_acc,epoch_loss)))

        y_pred = [0 if p < 0.5 else 1 for p in y_prob]
        print(confusion_matrix(y_true, y_pred))
        print(classification_report(y_true, y_pred))
        print('-------------')
            
    if resume: # 学習過程(Loss) を保存するか    
        plt.figure(figsize=(15,4))
        plt.plot(Loss['val'],label='val')
        plt.plot(Loss['train'],label='train')
        plt.legend()
        plt.savefig(os.path.join(output,'history.png'))
