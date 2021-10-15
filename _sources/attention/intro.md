Lab12: Sequence to Sequence Learning with Attention
=====================================================

In this lab we introduce the [basic attention (cross-attention)](https://weiliu2k.github.io/CITS4012/attention/attention.html) and the [self-attention mechansim](https://weiliu2k.github.io/CITS4012/attention/self_attention.html), applying them to the same square sequence predication problem we have seen in [Lab11's Sequence to Sequence Model](https://weiliu2k.github.io/CITS4012/LSTM/seq2seq.html).  

The second part of this lab focuses on a real neural machine translation task to build attention based models, one [with no sampling](https://weiliu2k.github.io/CITS4012/attention/NMT_No_Sampling.html) (i.e. using teacher forcing 100% of the time), and another with [scheduled sampling](https://weiliu2k.github.io/CITS4012/attention/NMT_scheduled_sampling.html) (i.e. using teacher forcing only sometimes). Since we have biGRU for the Encoder, the masked padded sequence in the backward layer would affect the results. So here we take the opportunity to introduce the [`PackedSequence`](https://weiliu2k.github.io/CITS4012/attention/PackedSequence.html) in PyTorch to handle variable length sequences.  

Credit: the notebooks are adapted from:

- Chpater 9 of [Deep Learning with PyTorch Step-by-Step](https://github.com/dvgodoy/PyTorchStepByStep)
- Chapter 8 of [Natural Language Processing with PyTorch](https://github.com/joosthub/PyTorchNLPBook/tree/master/chapters/chapter_8)