# Tensor Decomposition Algorithm for LDA Topic Models

Algorithm for estimating topics from LDA topic model using Tensor decomposition. The algorithm is based on robust tensor power iterations, and is described in the following papers.

1. Anandkumar, Animashree, et al. "A spectral algorithm for latent dirichlet allocation." Advances in Neural Information Processing Systems. 2012.

2. Anandkumar, Animashree, et al. "Tensor decompositions for learning latent variable models." The Journal of Machine Learning Research 15.1 (2014): 2773-2832.

## Usage

**TensorLDA.m** is the main function and has the following input/output.

```
Inputs:

 corpus = A data object having the following data input variables
    corpus.K = number of topics
    corpus.docs = (d \times n) matrix containing document samples drawn from LDA distribution. d = dimension/vocab size, n = no. of document samples
    corpus.alpha0 = hyper-parameter \alpha_0 of LDA distribution
 L = Number of random starts in tensor power iteration algo
 NumIter = max. number of tensor power iterations per component and per random start

Outputs:

 muMatHat = (d \times K) matrix containing the estimated topic vectors
 alphaArrHat = (1 \times K) matrix of the estimated topic fractions
```

Some example data is provided in the MAT file **ldaTest.mat**

It can be used to run the algorithm in MATLAB as follows.

```
>> load ldaTest
>> [muMatHat alphaArrHat] = TensorLDA(corpus,100,100);
```

It should print something like this...

```
Computing moments...
warning: blank docs = 0
Computing whitening matrix...
Performing robust tensor decomposition ...
Tensor power iteration for k = 1, Avg. power iter. = 0
Tensor power iteration for k = 2, Avg. power iter. = 11.08
Tensor power iteration for k = 3, Avg. power iter. = 11.03
Tensor power iteration for k = 4, Avg. power iter. = 11.14
Tensor power iteration for k = 5, Avg. power iter. = 11.04
Tensor power method complete!
```

Enjoy!