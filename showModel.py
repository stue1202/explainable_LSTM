import torch
from kan import KAN
import matplotlib.pyplot as plt
from myconstant import *

# 1. 載入完整的 KANLSTMModel 狀態字典
model_path = 'best_model_state_dict.pth'
state_dict = torch.load(model_path)

# 2. 提取 KAN 網路的狀態字典，並移除前綴
kan_state_dict_cell_0 = {}
for key, value in state_dict.items():
    print(f"Extracting key: {key}")
    if 'kan_activation.' in key and 'cells.0.' in key:
        
        new_key = key.replace('cells.0.kan_activation.', '')
        kan_state_dict_cell_0[new_key] = value

# 3. 創建一個「乾淨」的 KAN 實例，並立即進行剪枝
# 這樣可以確保內部狀態 (如 alpha_mask) 被正確建立
# 這裡我們將所有連接修剪到0% (因為我們只想要狀態變數，不真的要剪枝)
kan_model_0 = KAN([hidden_dim, hidden_dim])
print(kan_model_0)

# 4. 載入訓練好的權重
# 現在 kan_model_0 已經有了 plot() 所需的內部狀態，可以安全地載入權重了

# 5. 進行一次前向傳播，讓模型「看到」數據
x = torch.rand(30, 16) 
kan_model_0(x)

# 6. 現在可以安全地繪製圖形
print("正在繪製 KAN 網路...")
kan_model_0.plot()
plt.show()