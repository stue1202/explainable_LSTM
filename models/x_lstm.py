import torch
import torch.nn as nn
from torch.nn import Parameter
import math
from torchinfo import summary
from sklearn.preprocessing import MinMaxScaler
#from kan import *
from tools.myconstant import *
from .EfficientKAN import KAN
#from fastkan import *
mode='x_lstm'
class KANLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(KANLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
          
        # Separate parameters for each of the three standard gates (input, forget, output)
        self.weight_ih = Parameter(torch.Tensor(3 * hidden_size, input_size))
        self.weight_hh = Parameter(torch.Tensor(3 * hidden_size, hidden_size))
        self.bias_ih = Parameter(torch.Tensor(3 * hidden_size))
        self.bias_hh = Parameter(torch.Tensor(3 * hidden_size))

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

        # The KAN layer specifically for the cell_gate
        #self.kan_for_cell_gate = KAN([input_size + hidden_size, hidden_size])
        
        self.reset_parameters()
        self.kan_for_cell_gate = KAN([input_size + hidden_size, hidden_size])

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            if weight.data.ndimension() > 1:
                weight.data.uniform_(-stdv, stdv)

    def forward(self, input, hx):
        h_prev, c_prev = hx

        # 1. Calculate the linear components for the standard gates
        gates_linear = (torch.matmul(input, self.weight_ih.t()) + self.bias_ih +
                        torch.matmul(h_prev, self.weight_hh.t()) + self.bias_hh)

        input_gate_linear, forget_gate_linear, output_gate_linear = gates_linear.chunk(3, 1)

        input_gate = self.sigmoid(input_gate_linear)
        forget_gate = self.sigmoid(forget_gate_linear)
        output_gate = self.sigmoid(output_gate_linear)
        
        # 2. Use the KAN layer for the cell_gate
        # Concatenate the input and hidden state to feed into the KAN layer
        kan_input = torch.cat((input, h_prev), dim=1)
        cell_gate_kan_output = (self.kan_for_cell_gate(kan_input))

        # 3. Update the cell state (c_next) and hidden state (h_next)
        c_next = forget_gate * c_prev + input_gate * cell_gate_kan_output
        h_next = output_gate * self.tanh(c_next)
        
        return h_next, c_next

# The KANLSTMModel class remains the same
class X_LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(X_LSTM, self).__init__()
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