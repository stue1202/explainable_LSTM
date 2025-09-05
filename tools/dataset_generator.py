import numpy as np
import pandas as pd
import yfinance as yf
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from .myconstant import *
def create_sequences(data, seq_length,prediction_step):
        xs, ys = [], []
        for i in range(len(data) - seq_length -prediction_step +1):
            x = data[i:(i + seq_length)]
            y = data[(i + seq_length): i + seq_length+prediction_step,3]  # 預測未來 prediction_step 天的收盤價
            xs.append(x)
            ys.append(y)
        return np.array(xs), np.array(ys)
def dataset_split(seq_length,prediction_step,dataset):
    # 步驟1：下載更長的 S&P 500 指數數據
    # 為了確保所有資料集都有足夠的序列，我們下載到 2024 年底
    dataset_content = yf.download(dataset, start='2014-01-01', end='2025-01-01')
    features = ['Open', 'High', 'Low', 'Close', 'Volume']
    data = dataset_content[features].values

    # 步驟2：數據預處理與標準化
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    # 創建訓練數據集

    X, y = create_sequences(scaled_data, seq_length,prediction_step)

    # 根據日期進行資料集劃分
    # 訓練集：2014-2022
    train_end_date_str = '2023-01-01'
    val_end_date_str = '2024-01-01'
    
    # 找到最近的交易日
    train_end_date = pd.to_datetime(train_end_date_str)
    val_end_date = pd.to_datetime(val_end_date_str)

    # 使用 asof() 找到最接近且在日期之前的索引，然後再找到下一個索引
    #train_idx = dataset_content.index.asof(train_end_date)
    #val_idx = dataset_content.index.asof(val_end_date)
    
    train_end_idx = dataset_content.index.get_loc(dataset_content.index.asof(train_end_date))
    val_end_idx = dataset_content.index.get_loc(dataset_content.index.asof(val_end_date))

    # 確保切分點在序列長度之後
    train_end_seq_idx = train_end_idx - seq_length  
    val_end_seq_idx = val_end_idx - seq_length

    X_train = torch.from_numpy(X[:train_end_seq_idx]).float()
    y_train = torch.from_numpy(y[:train_end_seq_idx]).float()
    
    X_val = torch.from_numpy(X[train_end_seq_idx:val_end_seq_idx]).float()
    y_val = torch.from_numpy(y[train_end_seq_idx:val_end_seq_idx]).float()

    X_test = torch.from_numpy(X[val_end_seq_idx:]).float()
    y_test = torch.from_numpy(y[val_end_seq_idx:]).float()

    # 建立 DataLoader
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    print(f"train_size: {X_train.shape}, val_size: {X_val.shape}, test_size: {X_test.shape}")
    print(f'train_label size: {y_train.shape}')
    

    return train_loader, val_loader, test_loader, scaler