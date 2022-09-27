# Experiments with Classification Transformers

This is an independent self-study project where I tried to implement a classification transformer from scatch on Pytorch. A lot of the discussion here is based on previous blog posts or papers, such as [Peter Bloem's blog](http://peterbloem.nl/blog/transformers), [Kikaben](https://kikaben.com/transformers-training-details/), or [n8henrie](https://n8henrie.com/2021/08/writing-a-transformer-classifier-in-pytorch/). I also referred pretty extensively to Vaswani et al's [Attention is All You Need](https://arxiv.org/abs/1706.03762), and Coursera's [Sequence models course](https://www.coursera.org/learn/nlp-sequence-models). 

# Requirements and Installation and 

You need the basic PyTorch libraries (torch, torchtext, torchdata, torchmetrics, torchsummary). You can clone the repository and run install.sh if you don't have these requirements satisfied.

You also need the standard data science libraries in python (numpy, pandas, matplotlib).

# Running an experiment

To run the full set of experiments that I did, clone the repo annd run:

```

python code/experiments.py

```

You can also visualize the outputs of the experiments in Jupyter Notebook in the experiments/ folder. 