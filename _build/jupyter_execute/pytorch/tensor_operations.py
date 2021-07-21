Tensor Operations
=================

## Addition

import torch
from functions import describe
x = torch.randn(2,3)
describe(x)

describe(torch.add(x,x))

describe(x + x)

## Dimension based tensor operations

x = torch.arange(6)
describe(x)


x = x.view(2,3)
describe(x)

describe(torch.sum(x, dim=0))

describe(torch.sum(x, dim=1))

describe(torch.transpose(x, 0, 1))

