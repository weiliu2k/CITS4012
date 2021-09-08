# Lab06: Neural Network Building Blocks
Each node in a neural network is called a **perceptron** unit, which has three "knobs", a set of weights ($w$), a bias ($b$), and an activation function ($f$). The weights and bias are learned from the data, and the activation function is hand picked depending on the network designer's intuition of the network and its target outputs. Mathematically,

$y = f(wx + b)$

```{code-block} python
---
lineno-start: 1
emphasize-lines: 3, 11, 12, 23
caption:
    A skeleton of a Perceptron
---
class Perceptron(nn.Module):
    """
    A perceptron is one linear layer 
    """
    
    def __init__(self, input_dim):
        """
        Args:
            input_dim (int): size of the input features
        """
        super(Perceptron, self).__init__()
        self.fc1 = nn.Linear(input_dim, 1)

    def forward(self, x_in):
        """The forward pass of the perceptron

        Args:
            x_in (torch.Tensor): an input data tensor
                x_in.shape should be (batch, num_features)
        Returns:
            the resulting tensor. tensor.shape should be (batch,).
        """
        return torch.sigmoid(self.fc1(x_in))
```        
In this lab, we will look at the activation functions, and loss functions in more detail, and finish with building and train a simple neural network in PyTorch for simulating the XOR binary operator with Object-orientation in mind. 

Reference: [*Deep Learning in PyTorch, Step by Step A Beginner's Guide* by Daniel Voigt Godoy](https://leanpub.com/pytorch) - Chapter 2 and 3. 