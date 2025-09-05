#from tools.training import *
#
#from models.x_lstm import X_LSTM
#from models.LSTM_original import LSTM
#(Parameter_number,tarin_loss_list,train_r2_list,val_loss_list,val_r2_list,eval_loss,eval_r2,interval_time)=(train_model(28,8,2,1,"lstm",LSTM,'GC=F'))
#print(f"Parameter_number: {Parameter_number}\n eval_loss: {eval_loss}\n eval_r2: {eval_r2}\n interval_time: {interval_time}\n train_loss: {tarin_loss_list}\n val_loss: {val_loss_list}\n train_r2: {train_r2_list}\n val_r2: {val_r2_list}")
import os
if not os.path.exists("nn"):
    os.makedirs("nn")
test_data_path=os.path.join("nn","test.txt")
with open(test_data_path, 'x') as f:
    f.write("hello world".encode('utf-8'))