import pandas as pd
import yfinance as yf
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from torch import nn, optim
from .dataset_generator import dataset_split
from itertools import islice
import tqdm
import os
import time
import logging
from torchmetrics.regression import R2Score
from .myconstant import *
def train_model(seq_length, hidden_dim, num_layers, prediction_step,model_name,model_function,dataset_name,train_loader, val_loader, test_loader):
    
    stamp=str(seq_length)+"_"+str(hidden_dim)+"_"+str(num_layers)+"_"+str(prediction_step)+"_"+dataset_name+"_"+model_name
    saved_model_path=os.path.join("saved_model",stamp+".pth")
    tarin_loss_list=[]
    val_loss_list=[]
    train_r2_list=[]
    val_r2_list=[]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = model_function(input_dim, hidden_dim, prediction_step, num_layers).to(device)
    #model = StandardLSTMModel(input_dim, hidden_dim, output_dim, num_layers).to(device)
    criterion = nn.MSELoss()
    criterion2 = R2Score()
    optimizer = optim.Adam(model.parameters(), lr)
    best_val_loss=100
    patience_counter = 0
    # --- 訓練與驗證迴圈 ---
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        criterion2.reset()
        for X_train, y_train in tqdm.tqdm(train_loader):
            X_train = X_train.to(device)
            y_train = y_train.to(device)
            optimizer.zero_grad()
            outputs = model(X_train)
            loss = criterion(outputs, y_train)
            criterion2.update(outputs, y_train)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        avg_train_loss = train_loss / len(train_loader)
        train_r2 = criterion2.compute()

        tarin_loss_list.append(avg_train_loss)
        train_r2_list.append(train_r2.item())

        model.eval()
        val_loss = 0
        criterion2.reset()
        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val = X_val.to(device)
                y_val = y_val.to(device)
                outputs = model(X_val)
                #print("output",outputs.shape)
                loss = criterion(outputs, y_val)
                criterion2.update(outputs, y_val)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        val_r2 = criterion2.compute()
        val_loss_list.append(avg_val_loss)
        val_r2_list.append(val_r2.item())

        # 新增: Early Stopping 邏輯
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter=0
            torch.save(model.state_dict(), saved_model_path)
        else:
            patience_counter += 1
        if patience_counter >= patience:
            break

    #writer.close()

    # 儲存模型
    model.load_state_dict(torch.load(saved_model_path))

    # --- 最終測試 ---
    model.eval()
    with torch.no_grad():
        total_test_loss = 0
        criterion2.reset()
        for X_test, y_test in test_loader:
            start_time = time.time()
            outputs = model(X_test)
            end_time = time.time()
            interval_time = end_time - start_time
            loss = criterion(outputs, y_test)
            criterion2.update(outputs, y_test)
            total_test_loss += loss.item()
        test_loss = total_test_loss / len(test_loader)
        test_r2 = criterion2.compute()

    Parameter_number=0
    for name, param in model.named_parameters():
        Parameter_number+=param.numel()
    return Parameter_number, tarin_loss_list, train_r2_list, val_loss_list, val_r2_list, test_loss, test_r2, interval_time

