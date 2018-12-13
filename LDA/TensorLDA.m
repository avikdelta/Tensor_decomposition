function [muMatHat alphaArrHat] = TensorLDA(corpus,L,NumIter)

%-----------------------------------------------------------
% Author: Avik Ray (avik@utexas.edu)
% Copyright (C) 2015-2018 by Avik Ray
%
% Description: This script implements tensor decomposition
% algorithm based on tensor power iterations for recovering
% topics using documents/samples drawn from LDA distribution. The algoritm
% is described in Anandkumar et al. (2014).
%
% Inputs:
%
% corpus = A data object having the following data input variables
%    corpus.K = number of topics
%    corpus.docs = (d \times n) matrix containing document samples drawn
%    from LDA distribution. d = dimension/vocab size, n = no. of document samples
%    corpus.alpha0 = hyper-parameter \alpha_0 of LDA distribution
% L = Number of random starts in tensor power iteration algo
% NumIter = max. number of tensor power iterations per component and
% per random start
%
% Outputs:
%
% muMatHat = (d \times K) matrix containing the estimated topic vectors
% alphaArrHat = (1 \times K) matrix of the estimated topic fractions
%
% Example usage:
%
% [muMatHat alphaArrHat] = TensorLDA(corpus,100,200);
%-----------------------------------------------------------

% Init
K = corpus.K;
A = corpus.docs;
[d N] = size(A);
alpha0 = corpus.alpha0;

% Normalize columns
wordCounts = sum(A,1);

disp('Computing moments...');
X = A;
blankDocs = 0;
for i = 1:N
    if wordCounts(i)>0
        X(:,i) = X(:,i)/wordCounts(i);
    else
        blankDocs = blankDocs + 1;
    end
end
disp(['warning: blank docs = ' num2str(blankDocs)])
%X = A*diag(1./wordCounts);

% Compute mean
M = sum(X,2)/N;

% Compute pairs P = sum alpha_i/(alpha0*(1+alpha0))*mu_i*mu_i'
P = X*X'/N - (alpha0/(1+alpha0))*M*M';

% Whitening step 
disp('Computing whitening matrix...');
%[VFull D VFull1] = svd(P);
[V D V1] = svds(P,K);
%V = VFull(:,1:K);

Dhalf = zeros(K);
DhalfInv = zeros(K);
for k = 1:K
    Dhalf(k,k) = sqrt(D(k,k));
    DhalfInv(k,k) = 1/sqrt(D(k,k));
end

% Whitening matrix
W = DhalfInv*V';

% Tensor decomposition
disp('Performing robust tensor decomposition ...');
[lambdaArr, thetaMat] = RobustTensorPowerLDA(X,W,M,K,alpha0,L,NumIter);

% Compute mu
muMatHat = zeros(d,K);
for k = 1:K
    muMatHat(:,k) = ((alpha0+2)/2)*lambdaArr(k)*V*Dhalf*thetaMat(:,k);
end
alphaArrHat = (4*(alpha0+1)*alpha0/(alpha0+2)^2)*1./(lambdaArr.^2);

end