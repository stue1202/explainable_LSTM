from kan import *
import torch
import torch.nn as nn
import torch.optim as optim
from myconstant import *
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
criterion = nn.MSELoss()

f = lambda x: torch.exp(torch.sin(torch.pi*x[:,[0]]) + x[:,[1]]**2)
dataset = create_dataset(f, n_var=2, device=device)
dataset['train_input'].shape, dataset['train_label'].shape
model = KAN(width=[2,5,1], grid=5, k=3, seed=1, device=device)
optimizer = optim.Adam(model.parameters(), lr)
output=model(dataset['train_input'])

model.fit(dataset, opt="LBFGS", steps=20, lamb=0.01)
torch.save(model.state_dict(), 'kan.pth')
#for epoch in range(len(dataset['train_input'])):
#    output=model(dataset['train_input'][[epoch]])
#    loss=criterion(output, dataset['train_label'][[epoch]])
#    optimizer.zero_grad()
#    loss.backward()
#    optimizer.step()
#    print(f"epoch: {epoch}, loss: {loss.item()}")
model2=KAN(width=[2,5,1], grid=5, k=3, seed=1, device=device)
#model2.load_state_dict(torch.load('kan.pth'))
model2(dataset['train_input'])
print("finish")
model2.plot()
plt.show()