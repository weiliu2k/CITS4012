Types of Tensor
=================

## A helper function `describe(x)`
`x` is a torch tensor

NOTE: `tensor.shape` is a property, not a callable function

def describe(x):
    print("Type:{}".format(x.type()))
    print("Shape/size:{}".format(x.shape))
    print("Values: \n{}".format(x))


## Creating a tensor with `torch.Tensor()`

import torch

describe(torch.Tensor(2,3))

## Creating a randomly initialized tensor

import torch

describe(torch.rand(2,3))   # uniform random
describe(torch.randn(2,3))  # normal random

## Creating a filled tensor

import torch

describe(torch.zeros(2,3))

x = torch.ones(2,3)
describe(x)

x.fill_(5)
describe(x)

## Creating and initialising a tensor from lists

x = torch.Tensor([[1, 2, 3],
                  [4, 5, 6]])
describe(x)                  

## Creating and initialising a tensor from Numpy

import torch
import numpy as np

npy = np.random.rand(2,3)
describe(torch.from_numpy(npy))




```{toctree}
:hidden:
:titlesonly:
:numbered: True

create_tensors
tensor_operations
tensor_slicing
```
