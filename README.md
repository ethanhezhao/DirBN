# DirBN

The demo code of PFA+DirBN in the paper of "Dirichlet belief networks for topic structure learning", NIPS 2018 [Arxiv](https://arxiv.org/abs/1811.00717).

Key features:

1. DirBN discovers topic hierarchies on topic-word distributions.
2. DirBN flexibly combines with many other topic models.
3. DirBN enjoys better perplexity and topic coherence, especially for short texts.

# Run PFA+DirBN

0. The code is a mixture of Matlab and C++. The code has been tested in MacOS and Linux (Ubuntu). To run it on Windows, you need to re-compile all the .c files with MEX and a C++ complier.

1. Requirements: Matlab 2016b (or later).

2. We have offered the TMN dataset used in the paper, which is stored in MAT format, with the following contents:
- x: a V by N count (sparse) matrix for N documents with V words in the vocabulary
- voc: the words in the vocabulary
- train_idx: the indexes of documents for training 
- test_idx: the indexes of documents for testing

Please prepare your own documents in the above format. If you want to use this dataset, please cite the original papers, which are cited in our paper.

3. Run ```PFA_DirBN_demo.m```

# Use DirBN with other models

DirBN is a hierarchical construction on top of topic-word distributions and leaves the construction on doc-word distributions untouched. ```init_DirBN.m, sample_DirBN.m, sample_DirBN_beta.m, sample_DirBN_counts.m``` can be viewed as an independent package of DirBN. To combine DirBN with other topic models than PFA, simply call ```init_DirBN.m``` before the inference begins and call ```sample_DirBN.m``` in each iteration after the topic assignments are sampled.
# Notes

1. ```CRT_sum_mex_matrix_v1.c, CRT_sum_mex_v1.c, Mult_Sparse.c, Multrnd_Matrix_mex_fast_v1.c, PartitionX_v1.m, Sample_rk.m``` are borrowed from [GBN](https://github.com/mingyuanzhou/GBN) of [Mingyuan Zhou](https://mingyuanzhou.github.io). If you want to use the above code please cite the related papers. ```collapsed_gibbs_topic_assignment_mex.c``` is modified from the code of GBN. 

2. If you find any bugs, please contact me by email (ethanhezhao@gmail.com).
