import pandas as pd
import yfinance as yf
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from torch import nn, optim
from SP500_dataset import SP500_split
from myconstant import *
from itertools import islice
import tqdm
import os
from x_lstm import *
#from explainable_KAN import *
#from LSTM_origin import *
from timestamp import get_time_stamp
import logging

path=os.path.join('logs', get_time_stamp()+' '+mode)
logging.basicConfig(
    filename=path, # 日誌檔案名稱
    filemode='w',       # 'a' 代表追加模式，'w' 代表覆蓋模式
    level=logging.DEBUG, # 設定日誌級別為 DEBUG
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
# 訓練模型
#from torch.utils.tensorboard import SummaryWriter
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_loader, val_loader, test_loader, scaler = SP500_split(seq_length)
model = KANLSTMModel(input_dim, hidden_dim, output_dim, num_layers).to(device)
#model = StandardLSTMModel(input_dim, hidden_dim, output_dim, num_layers).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr)
best_val_loss=100
patience_counter = 0
# --- 訓練與驗證迴圈 ---
for epoch in range(epochs):
    model.train()
    train_loss = 0
    for X_train, y_train in tqdm.tqdm(train_loader):
        #print("X_train",X_train.shape)
        #print("y_train",y_train.shape)
        X_train = X_train.to(device)
        y_train = y_train.to(device)
        optimizer.zero_grad()
        outputs = model(X_train)
        #print("output",outputs.shape)
        loss = criterion(outputs, y_train)
        #l1_regularization = torch.tensor(0.0, device=X_train.device)
        #for name, param in model.named_parameters():
        #    if 'kan_activation.layers.0.spline_weight' in name:
        #        l1_regularization += torch.sum(torch.abs(param))
        #loss += lamb_l1 * l1_regularization
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    avg_train_loss = train_loss / len(train_loader)
    logging.info(f'Epoch [{epoch + 1}/{epochs}], Train Loss: {avg_train_loss:.4f}')
    # 新增: 驗證迴圈 (用於 Early Stopping)
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X_val, y_val in val_loader:
            X_val = X_val.to(device)
            y_val = y_val.to(device)
            outputs = model(X_val)
            #print("output",outputs.shape)
            loss = criterion(outputs, y_val)
            val_loss += loss.item()
    avg_val_loss = val_loss / len(val_loader)
    #best_val_loss= avg_val_loss if avg_val_loss<best_val_loss else best_val_loss
    logging.info(f'Epoch [{epoch + 1}/{epochs}], Validation Loss: {avg_val_loss:.4f}')

    # 新增: Early Stopping 邏輯
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter=0
        # 儲存最佳模型
        torch.save(model.state_dict(), 'best_model_state_dict.pth')
        logging.info("update best model")
    else:
        patience_counter += 1
        logging.info(f"patience_counter: {patience_counter}/{patience}")
    
    if patience_counter >= patience:
        logging.info("overfitting detected, stopped")
        break

#writer.close()

# 儲存模型
model.load_state_dict(torch.load('best_model_state_dict.pth'))

# --- 最終測試 ---
real_prices = []
predicted_prices = []
days=len(real_prices)
model.eval()
with torch.no_grad():
    total_test_loss = 0
    for X_test, y_test in test_loader:
        outputs = model(X_test)
        
        loss = criterion(outputs, y_test)
        total_test_loss += loss.item()
    avg_test_loss = total_test_loss / len(test_loader)
    logging.info(f'eval loss: {avg_test_loss:.4f}')



