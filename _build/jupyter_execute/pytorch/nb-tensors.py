def describe(x):
    print("Type:{}".format(x.type()))
    print("Shape/size:{}".format(x.shape))
    print("Values: \n{}".format(x))


import torch

describe(torch.Tensor(2,3))

