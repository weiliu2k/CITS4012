CUDA tensors
=============

import torch
from functions import describe

print(torch.cuda.is_available())

# prefered method: device agnostic tensor instantiation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


x = torch.rand(3,2).to(device)
describe(x)

```{warning}
Mixing CUDA tensors with CPU-bound tensors will lead to errors. This is because we need to ensure the tensors are on the same device. 
```

y = torch.rand(3,2)
x + y

cpu_device = torch.device("cpu")
x = x.to(cpu_device)
y = y.to(cpu_device)
x + y

```{note}
It is expensive to move data back and forth from the GPU. Best practice is to carry out as much computation on GPU as possible and then just transfering the final results to CPU. 
```