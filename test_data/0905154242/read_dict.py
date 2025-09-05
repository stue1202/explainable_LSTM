import ast
import os
import numpy as np


with open(os.path.join(os.getcwd(),"0905154242.txt"), "r") as f:
    s = f.read()

data = ast.literal_eval(s)

print(data)   # <class 'dict'>
k=data.values()
for list_ in k:
    _=np.array(list(list_.values()))
    
    print("f",(_))
    print(_.shape)