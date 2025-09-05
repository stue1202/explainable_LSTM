import torch
import torch.nn as nn
mode='original_pytorch_lstm'
class KANLSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(KANLSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # 使用 PyTorch 內建的 nn.LSTM
        # batch_first=True 表示輸入和輸出的維度順序為 (batch, sequence, features)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        
        # 全連接層，用於將 LSTM 的輸出轉換為最終的預測結果
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x 的維度為 (batch_size, sequence_length, input_dim)
        
        # 初始化隱藏狀態和單元狀態
        # h0 的維度為 (num_layers, batch_size, hidden_dim)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        # 將輸入 x 和初始狀態 (h0, c0) 傳入 LSTM
        # out: 所有的時間步輸出，其維度為 (batch, sequence, hidden_dim)
        # (hn, cn): 最後一個時間步的隱藏狀態和單元狀態
        out, (hn, cn) = self.lstm(x, (h0, c0))
        
        # 只取最後一個時間步的輸出，傳入全連接層
        out = self.fc(out[:, -1, :])
        
        return out