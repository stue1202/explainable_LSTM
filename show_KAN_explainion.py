import torch
import os
# 載入你已儲存的狀態字典
# 請將 'kan_lstm_model_state_dict.pth' 替換成你的實際檔案路徑
try:
    state_dict = torch.load('kan_lstm_model_state_dict.pth')
    
    print("--- 模型的每一層名稱 (Keys) ---")
    # 使用 .keys() 方法來獲取所有層次的名稱，並逐一列印
    for key in state_dict.keys():
        print(key)

except FileNotFoundError:
    print("錯誤：找不到模型檔案 'kan_lstm_model_state_dict.pth'。請確認檔案路徑是否正確。")
