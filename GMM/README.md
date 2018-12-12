# Tensor Decomposition Algorithm for GMM

Algorithms for estimating means of a Gaussian mixture model using Tensor decomposition. The algorithm is based on robust tensor power iterations, and is described in the following papers.

1. Hsu, Daniel, and Sham M. Kakade. "Learning mixtures of spherical gaussians: moment methods and spectral decompositions." Proceedings of the 4th conference on Innovations in Theoretical Computer Science. ACM, 2013.

2. Anandkumar, Animashree, et al. "Tensor decompositions for learning latent variable models." The Journal of Machine Learning Research 15.1 (2014): 2773-2832.

## Usage

**TensorGMM.m** is the main function and has the following input/output.

```
Inputs:

 data = A data object having the following data input variables
    data.K = number of GMM components
    data.samples = (d \times n) matrix containing samples drawn from GMM distribution. d = dimension, n = no. of samples
 L = Number of random starts in tensor power iteration algo
 NumIter = max. number of tensor power iterations per component and per random start

Outputs:

 muMatHat = (d \times K) matrix containing the estimated Gaussian mean vectors
 alphaArrHat = (1 \times K) matrix of the estimated Gaussian fractions
```

Some example data is provided in the MAT file **dataTest.mat**

It can be used to run the algorithm in MATLAB as follows.

```
>> load dataTest
>> [muMatHat alphaArrHat] = TensorGMM(dataTest,100,100);
```

It should print something like this...

```
Performing robust tensor decomposition ...
Tensor power iteration for k = 1, Avg. power iter. = 0
Tensor power iteration for k = 2, Avg. power iter. = 11.17
Tensor power iteration for k = 3, Avg. power iter. = 11.12
Tensor power iteration for k = 4, Avg. power iter. = 11.04
Tensor power iteration for k = 5, Avg. power iter. = 11.02
Tensor power iteration for k = 6, Avg. power iter. = 11.18
Tensor power iteration for k = 7, Avg. power iter. = 11.09
Tensor power iteration for k = 8, Avg. power iter. = 11.07
Tensor power iteration for k = 9, Avg. power iter. = 11.1
Tensor power iteration for k = 10, Avg. power iter. = 11
Tensor power method complete!
```

Enjoy!