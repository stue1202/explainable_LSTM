from tools.training import *
from models.x_lstm import X_LSTM
from models.LSTM_original import LSTM
import pandas as pd
from tools.timestamp import get_time_stamp
import numpy as np
# 建立資料
csv = {
    "seq_length": [],
    "hidden_dim": [],
    "num_layers": [],
    "prediction_step": [],
    "Dataset": [],
    "model_name": [],
    "MSE": [],
    "R^2": [],
    "params_numbers": [],
    "interval_time": [],
}
tmp_series = dict()
params = {
    "batch_size": 30,
    "seq_length": [30],
    "input_dim": 5,
    "hidden_dim": [8,16,32],
    "num_layers": [1],
    "epochs": 100,
    "lr": 0.001,
    "lamb_l1": 0.1,
    "patience": 5,
    "prediction_step": [1,3,5],
    "dataset": ['BTC-USD','GC=F','^GSPC'],
    "average_eval_time":1,
    "model":[("x_lstm",X_LSTM),("original_lstm",LSTM)]
}
def eliminate_outliers(data):
    data = np.array(data)
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    # 過濾掉極端值
    filtered = data[(data >= lower_bound) & (data <= upper_bound)]
    return np.mean(filtered)

def find_min_len(data):
    return min(len(row) for row in data)
def avarage_lists(data):
    min_len=find_min_len(data)
    # 截斷每列到最短長度，轉成 tensor
    tensors = [torch.tensor(row[:min_len]) for row in data]

    # 對應元素相加
    sum_tensor = torch.stack(tensors).sum(dim=0)
    sum_tensor = sum_tensor / params["average_eval_time"]
    return sum_tensor.tolist()

    
            
def run_experiment(params):
    now_time=get_time_stamp()
    for dataset_name in params["dataset"]:
        for seq_length in params["seq_length"]:
            for prediction_step in params["prediction_step"]:
                train_loader, val_loader, test_loader, scaler = dataset_split(seq_length,prediction_step,dataset_name)
                for hidden_dim in params["hidden_dim"]:
                    for num_layers in params["num_layers"]:
                        for model_name,model_function in params["model"]:
                            stamp=str(seq_length)+"_"+str(hidden_dim)+"_"+str(num_layers)+"_"+str(prediction_step)+"_"+dataset_name+"_"+model_name
                            train_mse_collection=[];train_r2_collection=[];val_mse_collection=[];val_r2_collection=[];eval_mse_collection=[];eval_r2_collection=[];interval_time_collection=[]
                            csv["seq_length"].append(seq_length)
                            csv["hidden_dim"].append(hidden_dim)
                            csv["num_layers"].append(num_layers)
                            csv["prediction_step"].append(prediction_step)
                            csv["Dataset"].append(dataset_name)
                            csv["model_name"].append(model_name)
                            # average
                            for _ in range(params["average_eval_time"]):
                                (Parameter_number,tarin_loss_list,train_r2_list,val_loss_list,val_r2_list,eval_loss,eval_r2,interval_time)=train_model(seq_length,hidden_dim, num_layers, prediction_step,model_name,model_function,dataset_name,train_loader, val_loader, test_loader)
                                train_mse_collection.append(tarin_loss_list)
                                train_r2_collection.append(train_r2_list)
                                val_mse_collection.append(val_loss_list)
                                val_r2_collection.append(val_r2_list)
                                eval_mse_collection.append(eval_loss)
                                eval_r2_collection.append(eval_r2)
                                interval_time_collection.append(interval_time)
                            #list
                            tmp_series[stamp]={
                                "train_mse":avarage_lists(train_mse_collection),
                                "train_r2":avarage_lists(train_r2_collection),
                                "val_mse":avarage_lists(val_mse_collection),
                                "val_r2":avarage_lists(val_r2_collection)
                            }
                            #scalar
                            csv["MSE"].append(eliminate_outliers(eval_mse_collection))
                            csv['R^2'].append(eliminate_outliers(eval_r2_collection))
                            csv['interval_time'].append(eliminate_outliers(interval_time_collection))
                            csv['params_numbers'].append(Parameter_number)
                            #draw pictures
                            print(f"Start training: seq_length={seq_length}, hidden_dim={hidden_dim}, num_layers={num_layers}, prediction_step={prediction_step}, dataset={dataset_name}, model={model_name}")
                            print("-----------------------------------------------------"   )
    import os
    # picture
    test_data_path=os.path.join("test_data",now_time)
    if not os.path.exists(test_data_path):
        os.makedirs(test_data_path)
    test_data_loss_path=os.path.join(test_data_path,now_time+".txt")
    # csv
    with open(test_data_loss_path, 'x') as f:
        f.write(str(tmp_series))
    

    csv_path=os.path.join(test_data_path,now_time+".csv")
    df = pd.DataFrame(csv)
    df.to_csv(csv_path, index=False)

run_experiment(params)
                            






                                


