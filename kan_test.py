from kan import *
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

model = KAN([2,2])
x = torch.normal(0,1,size=(100,2)).to(device)
# plot KAN at initialization
model(x)
model.plot() 
plt.show()
