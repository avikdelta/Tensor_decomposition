function [muMatHat alphaArrHat] = TensorGMM(data,L,NumIter)

%-----------------------------------------------------------
% Author: Avik Ray (avik@utexas.edu)
% Copyright (C) 2015-2018 by Avik Ray
%
% Description: This script implements tensor decomposition
% algorithm based on tensor power iterations for recovering
% components of a Gaussian Mixture Model as described in
% Hsu and Kakade (2013), Anandkumar et al. (2014).
%
% Inputs:
%
% data = A data object having the following data input variables
%    data.K = number of GMM components
%    data.samples = (d \times n) matrix containing samples drawn
%    from GMM distribution. d = dimension, n = no. of samples
% L = Number of random starts in tensor power iteration algo
% NumIter = max. number of tensor power iterations per component and
% per random start
%
% Outputs:
%
% muMatHat = (d \times K) matrix containing the estimated Gaussian mean
% vectors
% alphaArrHat = (1 \times K) matrix of the estimated Gaussian fractions
%
% Example usage:
%
%[muMatHat alphaArrHat] = TensorGMM(data,100,100);
%-----------------------------------------------------------

% Init
K = data.K;
X = data.samples;
[d N] = size(X);

% Compute mean
xbar = sum(X,2)/N;

% Compute covariance
Cov = X*X'/N - xbar*xbar';

% Estimate sigmasqr
[Vc Sc Vc1] = svd(Cov);
sigmasqr = Sc(d,d);

% Estimate M1
Xmean = repmat(xbar,1,N);
v = Vc(:,d);
coeff = v'*(X-Xmean);
M1 = X*(coeff.^2)'/N;

% Estimate M2
M2 = X*X'/N - sigmasqr*eye(d);

% Whitening step 
[VFull D VFull1] = svd(M2);
V = VFull(:,1:K);

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
[lambdaArr, thetaMat] = RobustTensorPowerGMM(X,W,M1,K,L,NumIter);

% Compute muhat
muMatHat = zeros(d,K);
for k = 1:K
    muMatHat(:,k) = lambdaArr(k)*V*Dhalf*thetaMat(:,k);
end
alphaArrHat = 1./(lambdaArr.^2);

end