Lab06: Neural Network Building Blocks
======================================
## Perceptron

Each node in a neural network is called a perceptron unit, which has three "knobs", a set of weights ($w$), a bias ($b$), and an activation function ($f$). The weights and bias are learned from the data, and the activation function is hand picked depending on the network designer's intuition of the network and its target outputs. Mathematically,

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
        return torch.sigmoid(self.fc1(x_in)).squeeze()

```        

```{image} ../images/nlp_pytorch_book.jpg
:alt: Pytorch for NLP Book
:class: bg-primary mb-1
:width: 200px
:align: left
```
```{image} ../images/logo_pytorch.jpeg
:alt: Pytorch Logo
:class: bg-primary mb-1
:width: 100px
:align: right
```
Reference: *Natural Lanuage Processing with PyTorch* - Building intelligent lanaguage applications using deep learning, by Delip Rao and Brian McMahan (copyright O'REILLY Feb 2019)