Lab11: GRU, LSTM and Seq2Seq
==================================

In this lab we introduce two more powerful variants of RNNs, namely, [Gated Recurrent Units (GRUs)](https://weiliu2k.github.io/CITS4012/LSTM/gru.html) and [Long Short Term Memorys (LSTMs)](https://weiliu2k.github.io/CITS4012/LSTM/lstm.html). We then [compare their performance with Elman RNN](https://weiliu2k.github.io/CITS4012/LSTM/gru_lstm_square.html) in the same square sequence direction binary classification task. 

The second part of this lab focuses on the Sequence to Sequence models, first introduced by [Ilya Sutskever, Oriol Vinyals, Quoc V. Le](https://proceedings.neurips.cc/paper/2014/file/a14ac55a4f27472c5d894ec1c3c743d2-Paper.pdf) from Google at NeurIPS 2014. Then we will forcus on the decoder part for a Surname Generation task [unconditioned](https://weiliu2k.github.io/CITS4012/LSTM/Surname_Generation_Unconditioned.html) and [conditioned](https://weiliu2k.github.io/CITS4012/LSTM/Surname_Generation_Conditioned.html).

Credit: the notebooks are adapted from:

- Chpater 8 of [Deep Learning with PyTorch Step-by-Step](https://github.com/dvgodoy/PyTorchStepByStep)
- Chpater 9 of [Deep Learning with PyTorch Step-by-Step](https://github.com/dvgodoy/PyTorchStepByStep)
- Chapter 7 of [Natural Language Processing with PyTorch](https://github.com/joosthub/PyTorchNLPBook/tree/master/chapters/chapter_7)