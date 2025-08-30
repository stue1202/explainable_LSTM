import pandas as pd
import yfinance as yf
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from modify.explainable_KAN import KANLSTMModel
from myconstant import *
# 常數設定

def SP500_split(seq_length):
    # 步驟1：下載更長的 S&P 500 指數數據
    print("正在下載 S&P 500 指數數據...")
    # 為了確保所有資料集都有足夠的序列，我們下載到 2024 年底
    sp500_data = yf.download('^GSPC', start='2014-01-01', end='2025-01-01')
    data = sp500_data['Close'].values.reshape(-1, 1)

    # 步驟2：數據預處理與標準化
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    # 創建訓練數據集
    def create_sequences(data, seq_length):
        xs, ys = [], []
        for i in range(len(data) - seq_length):
            x = data[i:(i + seq_length)]
            y = data[i + seq_length]
            xs.append(x)
            ys.append(y)
        return np.array(xs), np.array(ys)

    X, y = create_sequences(scaled_data, seq_length)

    # 根據日期進行資料集劃分
    train_end_date = pd.to_datetime('2023-01-01')
    val_end_date = pd.to_datetime('2024-01-01')
    
    # 找到最近的交易日索引，避免 Key Error
    train_idx_loc = sp500_data.index.get_loc(sp500_data.index.asof(train_end_date))
    val_idx_loc = sp500_data.index.get_loc(sp500_data.index.asof(val_end_date))

    train_end_seq_idx = train_idx_loc - seq_length
    val_end_seq_idx = val_idx_loc - seq_length
    
    # 訓練集
    X_train = torch.from_numpy(X[:train_end_seq_idx]).float()
    y_train = torch.from_numpy(y[:train_end_seq_idx]).float()
    # 驗證集
    X_val = torch.from_numpy(X[train_end_seq_idx:val_end_seq_idx]).float()
    y_val = torch.from_numpy(y[train_end_seq_idx:val_end_seq_idx]).float()
    # 測試集
    X_test = torch.from_numpy(X[val_end_seq_idx:]).float()
    y_test = torch.from_numpy(y[val_end_seq_idx:]).float()

    print(f"訓練序列數: {len(X_train)}")
    print(f"驗證序列數: {len(X_val)}")
    print(f"測試序列數: {len(X_test)}")

    # 建立 DataLoader
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 返回所有需要的物件
    return train_loader, val_loader, test_loader, scaler, X_train, y_train, X_test, y_test, sp500_data, train_end_seq_idx, val_end_seq_idx

# ----------------- 主訓練與評估流程 -----------------
# 載入資料並取得相關物件
(train_loader, val_loader, test_loader, scaler, 
 X_train, y_train, X_test, y_test, sp500_data, 
 train_end_seq_idx, val_end_seq_idx) = SP500_split(seq_length)

# 實例化 KANLSTMModel
model = KANLSTMModel(input_dim, hidden_dim, output_dim, num_layers)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 訓練與驗證迴圈
for epoch in range(epochs):
    model.train()
    total_train_loss = 0
    for X_train_batch, y_train_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_train_batch)
        loss = criterion(outputs, y_train_batch)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
    avg_train_loss = total_train_loss / len(train_loader)

    # 驗證階段
    model.eval()
    with torch.no_grad():
        total_val_loss = 0
        for X_val_batch, y_val_batch in val_loader:
            outputs = model(X_val_batch)
            loss = criterion(outputs, y_val_batch)
            total_val_loss += loss.item()
        avg_val_loss = total_val_loss / len(val_loader)

    print(f'Epoch [{epoch + 1}/{epochs}], Avg Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}')

# 最終測試與預測
model.eval()
with torch.no_grad():
    # 預測訓練集
    train_preds = model(X_train)
    # 預測測試集
    test_preds = model(X_test)
    
    # 計算測試集損失
    test_loss = criterion(test_preds, y_test)
    print(f'\n最終測試集上的平均損失: {test_loss.item():.4f}')

# 反正規化，將預測值還原回原始股價
train_preds = scaler.inverse_transform(train_preds.numpy())
y_train_orig = scaler.inverse_transform(y_train.numpy())
test_preds = scaler.inverse_transform(test_preds.numpy())
y_test_orig = scaler.inverse_transform(y_test.numpy())

# ----------------- 使用 Matplotlib 繪圖 -----------------
# 找到正確的日期索引
train_index = sp500_data.index[seq_length:train_end_seq_idx + seq_length]
test_index = sp500_data.index[val_end_seq_idx + seq_length:]

plt.figure(figsize=(15, 6))

# 繪製訓練集結果
plt.plot(train_index, y_train_orig, label='實際訓練股價', color='blue')
plt.plot(train_index, train_preds, label='預測訓練股價', color='lightblue', linestyle='--')

# 繪製測試集結果
plt.plot(test_index, y_test_orig, label='實際測試股價', color='red')
plt.plot(test_index, test_preds, label='預測測試股價', color='salmon', linestyle='--')

plt.title('S&P 500 指數預測', fontsize=16)
plt.xlabel('日期', fontsize=12)
plt.ylabel('收盤價', fontsize=12)
plt.legend()
plt.grid(True)
plt.show()