import pandas as pd
import yfinance as yf
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from torch import nn, optim
from tools.dataset_generator import dataset_split
from itertools import islice
import tqdm
import os
import time
from tools.timestamp import get_time_stamp
import logging
from torchmetrics.regression import R2Score
def train_model(batch_size, seq_length, input_dim, hidden_dim, output_dim, num_layers, epochs, lr, lamb_l1, patience, prediction_step,model_select,mode,dataset_name):
    model_name=os.path.join('saved_model', get_time_stamp()+' '+mode+'.pth')
    logs_name=os.path.join('logs', get_time_stamp()+' '+mode+'.txt')
    logging.basicConfig(
        filename=logs_name, # 日誌檔案名稱
        filemode='w',       # 'a' 代表追加模式，'w' 代表覆蓋模式
        level=logging.DEBUG, # 設定日誌級別為 DEBUG
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    # 訓練模型
    #from torch.utils.tensorboard import SummaryWriter
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, val_loader, test_loader, scaler = dataset_split(seq_length,prediction_step,dataset_name)
    model = model_select(input_dim, hidden_dim, prediction_step, num_layers).to(device)
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
            #l1_regularization = torch.tensor(0.0, device=X_train.device)
            #for name, param in model.named_parameters():
            #    if 'kan_activation.layers.0.spline_weight' in name:
            #        l1_regularization += torch.sum(torch.abs(param))
            #loss += lamb_l1 * l1_regularization
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        avg_train_loss = train_loss / len(train_loader)
        train_r2 = criterion2.compute()
        logging.info(f'Epoch [{epoch + 1}/{epochs}], Train Loss: {avg_train_loss:.4f}, r2: {train_r2.item():.4f}')
        # 新增: 驗證迴圈 (用於 Early Stopping)
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
        logging.info(f'Epoch [{epoch + 1}/{epochs}], Validation Loss: {avg_val_loss:.4f}, r2: {val_r2.item():.4f}')

        # 新增: Early Stopping 邏輯
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter=0
            # 儲存最佳模型
            torch.save(model.state_dict(), model_name)
            logging.info("update best model")
        else:
            patience_counter += 1
            logging.info(f"patience_counter: {patience_counter}/{patience}")
        
        if patience_counter >= patience:
            logging.info("overfitting detected, stopped")
            break

    #writer.close()

    # 儲存模型
    model.load_state_dict(torch.load(model_name))

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
        avg_test_loss = total_test_loss / len(test_loader)
        test_r2 = criterion2.compute()
        logging.info(f'eval loss: {avg_test_loss:.4f}, r2: {test_r2.item():.4f}, time per batch: {interval_time:.6f} seconds')

    logging.info(f'----print model parameters----')
    logging.info(f'using constant: {mode}')
    Parameter_number=0
    for name, param in model.named_parameters():
        logging.info(f"layer: {name}, parameters number: {param.numel()}")
        Parameter_number+=param.numel()
    logging.info(f"all arameters number: {Parameter_number}")
    return Parameter_number, avg_train_loss, train_r2.item(), avg_val_loss, val_r2.item(), avg_test_loss, test_r2.item()

