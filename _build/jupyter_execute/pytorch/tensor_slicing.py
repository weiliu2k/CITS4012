Tensor Slicing, Indexing and Joining
====================================

import torch
from functions import describe

x = torch.arange(6).view(2,3)
describe(x)

## Contiguous Indexing using `[:a, :b]`

The code below accesses up to row 1 but not including row 1, and up to col 2, but no including col 2.

describe(x[:1, :2])

## Noncontiguous Indexing

Using function `torch.index_select()`, the code below accesses column (`dim=1`) indexed by 0 and 2. 

indices = torch.LongTensor([0, 2])
describe(torch.index_select(x, dim=1, index=indices))

You can duplicate the same row or column multiple times, by specifying the same index multiple times. 

indices = torch.LongTensor([0, 0, 0])
describe(torch.index_select(x, dim=0, index=indices))

Use indices directly `[inices_list, indices_list]` can also achieve the same outcome.

row_indices = torch.arange(2).long()
col_indices = torch.LongTensor([0,2])
describe(x[row_indices, col_indices])

describe(x[[0,1], [0,2]])

## Concatenating Tensors

x = torch.arange(6).view(2,3)
describe(x)

describe(torch.cat([x, x], dim=0))

describe(torch.cat([x, x], dim=1))

describe(torch.stack([x, x], dim=1))

## Linear Algebra on tensors: multiplication

x1 = torch.arange(6).view(2,3).float()
describe(x1)

x2 = torch.ones(3,2)
x2[:, 1] += 1
describe(x2)

describe(torch.mm(x1, x2))

```{warning}
`torch.arange()` creates LongTensor, for `torch.mm()`, we need to convert the LongTensor to FloatTensor by using `x.float()`.
```