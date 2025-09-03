import pandas as pd
import yfinance as yf
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from explainable_KAN import KANLSTMModel # 請確保這個模型定義在你的檔案中
from torch import nn, optim
from SP500_dataset import SP500_split
from myconstant import *
# 訓練模型
#from torch.utils.tensorboard import SummaryWriter
#writer = SummaryWriter('runs/kan_lstm_exp')
train_loader, val_loader, test_loader, scaler = SP500_split(seq_length, input_dim)
model = KANLSTMModel(input_dim, hidden_dim, output_dim, num_layers)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr)

# --- 訓練與驗證迴圈 ---
for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")
    model.train()
    train_loss = 0
    now=0
    for X_train, y_train in train_loader:
        print(f"batch {now}")
        now+=1
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        l1_regularization = torch.tensor(0., requires_grad=True)
        for name, param in model.named_parameters():
            if 'kan_activation.layers.0.spline_weight' in name:
                l1_regularization = l1_regularization + torch.sum(torch.abs(param))
        loss += lamb_l1 * l1_regularization
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    avg_train_loss = train_loss / len(train_loader)

    #驗證階段
    #model.eval()
    #with torch.no_grad():
    #    val_loss = 0
    #    for X_val, y_val in val_loader:
    #        outputs = model(X_val)
    #        loss = criterion(outputs, y_val)
    #        val_loss += loss.item()
    #    avg_val_loss = val_loss / len(val_loader)

    print(f'Epoch [{epoch + 1}/{epochs}], Train Loss: {avg_train_loss:.4f}')
#writer.close()

# 儲存模型
torch.save(model.state_dict(), 'kan_lstm_model_state_dict.pth')
print("模型已成功儲存！")
# --- 最終測試 ---
model.eval()
with torch.no_grad():
    total_test_loss = 0
    for X_test, y_test in test_loader:
        outputs = model(X_test)
        loss = criterion(outputs, y_test)
        total_test_loss += loss.item()
    avg_test_loss = total_test_loss / len(test_loader)
    print(f'eval loss: {avg_test_loss:.4f}')