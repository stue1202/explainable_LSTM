from x_lstm import *
import torch
import torch.nn as nn
from myconstant import *
test_output = torch.randn(30,1)
test_output = test_output.unsqueeze(-1)
print(test_output)
#model = KANLSTMModel(input_dim, hidden_dim, output_dim, num_layers)
#output = model(test_output)
#print(test_output.shape, output.shape)