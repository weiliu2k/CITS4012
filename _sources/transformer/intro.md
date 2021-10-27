Extra: Transformers
======================

This set of notebooks first introduces the key elements for building a transformer encoder-decoder modle, namely, [multi-headed narrow attention](https://weiliu2k.github.io/CITS4012/transformer/transformer.html#narrow-attention), [layer normalisation](https://weiliu2k.github.io/CITS4012/transformer/LayerNorm.html), and [position encoding](https://weiliu2k.github.io/CITS4012/transformer/position_encoding.html). 

Then finally, we assemble them together to build our own Transformer Model, and compare it with the PyTorch `nn.Tranformer` class in the [transform and roll out notebook](https://weiliu2k.github.io/CITS4012/transformer/transformer.html). The picture below provides a good mapping of classes we implemented to the components used in a Transformer Encoder-Decoder Model. 

![Full Transformer with Class Mapping](../images/full_transformer_and_class.png)

Credit: the notebooks are adapted from:

- Chpater 9 of [Deep Learning with PyTorch Step-by-Step](https://github.com/dvgodoy/PyTorchStepByStep/blob/master/Chapter09.ipynb)
- Chpater 10 of [Deep Learning with PyTorch Step-by-Step](https://github.com/dvgodoy/PyTorchStepByStep/blob/master/Chapter10.ipynb)

