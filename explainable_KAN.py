import torch
import torch.nn as nn
from torch.nn import Parameter
import math
from torchinfo import summary
from sklearn.preprocessing import MinMaxScaler
from kan import *
from myconstant import *
#from EfficientKAN import KAN
class KANLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(KANLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
          
        # LSTM 門控機制中的權重
        self.weight_ih = Parameter(torch.Tensor(4 * hidden_size, input_size))
        self.weight_hh = Parameter(torch.Tensor(4 * hidden_size, hidden_size))
        self.bias_ih = Parameter(torch.Tensor(4 * hidden_size))
        self.bias_hh = Parameter(torch.Tensor(4 * hidden_size))

        self.sigmoid = nn.Sigmoid()
        self.kan_activation = KAN([hidden_size,hidden_size])
        
        self.tanh = nn.Tanh()
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, input, hx):
        # hx 是 (h_prev, c_prev)
        h_prev, c_prev = hx

        # 1. 計算所有門的輸入
        gates = (torch.matmul(input, self.weight_ih.t()) + self.bias_ih +
                 torch.matmul(h_prev, self.weight_hh.t()) + self.bias_hh)

        input_gate, forget_gate, cell_gate, output_gate = gates.chunk(4, 1)

        input_gate = self.sigmoid(input_gate)
        forget_gate = self.sigmoid(forget_gate)
        output_gate = self.sigmoid(output_gate)
        kan_output = self.kan_activation(cell_gate)
        
        c_next = forget_gate * c_prev + input_gate * kan_output
        h_next = output_gate * self.tanh(c_next)
        
        return h_next, c_next
class KANLSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(KANLSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.cells = nn.ModuleList([KANLSTMCell(input_dim if i == 0 else hidden_dim, hidden_dim) for i in range(num_layers)])
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        batch_size = x.size(0)
        h_t = [torch.zeros(batch_size, self.hidden_dim).to(x.device) for _ in range(self.num_layers)]
        c_t = [torch.zeros(batch_size, self.hidden_dim).to(x.device) for _ in range(self.num_layers)]

        for t in range(x.size(1)):
            input_t = x[:, t, :]
            for layer in range(self.num_layers):
                h_t[layer], c_t[layer] = self.cells[layer](input_t, (h_t[layer], c_t[layer]))
                input_t = h_t[layer]
        out = self.fc(h_t[-1])
        return out
