import torch
import torch.nn as nn
import torch.nn.functional as F

class U_t_train(nn.Module):
    """
    u(t) = 1 非発話
    　　    0 発話
    """
    def __init__(self, num_layers = 1, input_size=256, hidden_size = 32):
        super(U_t_train, self).__init__()
        
        self.fc1 = nn.Linear(input_size, hidden_size)
        #self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.lstm = torch.nn.LSTM(
            input_size = hidden_size, #入力size
            hidden_size = hidden_size, #出力size
            batch_first = True, # given_data.shape = (batch , frames , input_size)
        )
        self.fc3 = nn.Linear(hidden_size, 2)
        self.dr1 = nn.Dropout()
        self.relu1 = nn.ReLU()
        self.dr2 = nn.Dropout()
        self.relu2 = nn.ReLU()        

        self.hidden_size = hidden_size
        self.num_layers = 1
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, x):
        assert len(x.shape) == 3 , print('data shape is incorrect.')
        bs,fr,hs = x.size()
        x = self.dr1(self.relu1(self.fc1(x.view(-1,hs))))
        x = x.view(bs,fr,-1)
        h,_ = self.lstm(x,self.reset_state(bs))
        y = self.fc3(h[:,-1,:])

        return y

    def reset_state(self,bs):
        self.h = torch.zeros(self.num_layers, bs, self.hidden_size).to(self.device)
        return (self.h,self.h)

class TimeActionPredict(nn.Module):
    """
    行動予測するネットワーク
    """
    def __init__(self, input_size=256, hidden_size = 64):
        super(TimeActionPredict, self).__init__()
        
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.dr1 = nn.Dropout()
        self.relu1 = nn.ReLU()
        
        self.lstm = torch.nn.LSTM(
            input_size = hidden_size, #入力size
            hidden_size = hidden_size, #出力size
            batch_first = True, # given_data.shape = (batch , frames , input_size)
        )
        
        self.fc2 = nn.Linear(hidden_size, 2)        

        self.num_layers = 1
        self.hidden_size = hidden_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, x, hidden=None):
        
        assert len(x.shape) == 2 , print('data shape is incorrect.')
        
        x = self.dr1(self.relu1(self.fc1(x))) ## 2
        x = x.view(1,1,-1) # 2 -> 3
        if hidden is None:
            hidden = self.reset_state()
            #print('reset state!!')
        
        h, hidden = self.lstm(x, hidden)
        y = self.fc2(h[:,-1,:]) # (bs, frames, hidden_size) -> (bs, hidden_size)
        return y, hidden

    def reset_state(self):
        self.h = torch.zeros(self.num_layers, 1, self.hidden_size).to(self.device)
        return (self.h,self.h)
