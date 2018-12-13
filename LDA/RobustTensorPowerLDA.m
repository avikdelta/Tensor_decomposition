function [lambdaArr, thetaMat] = RobustTensorPowerLDA(X,W,M,K,alpha0,L,NumIter)

%-----------------------------------------------------------
% Author: Avik Ray (avik@utexas.edu)
% Copyright (C) 2015-2018 by Avik Ray
%
% Description: This script implements the main tensor power iteration
% routine for finding topics/components of a LDA distribution.
% [ref: Anandkumar et al. (2014)].
%
% Inputs:
%
% X = (d \times n) sample matrix
% W = (k \times d) whitening matrix
% M = (d \times 1) normalized first moment 
% K = number of topics/LDA components
% alpha0 = hyper-parameter \alpha_0 of LDA 
% L = Number of random starts in tensor power iteration algo
% NumIter = max. number of tensor power iterations per component and
% per random start
%
% Outputs:
%
% lambdaArr = array of K tensor eigenvalues
% thetaMat = (K \times K) matrix of K robust tensor eigenvectors
%
%-----------------------------------------------------------

% Init
[d N] = size(X);
lambdaArr = [];
thetaMat = [];
epsilon = 1e-3;
Xtilde = W*X;
Mtilde = W*M;
avgPowerIter = 0;

for k = 1:K
    disp(['Tensor power iteration for k = ' num2str(k) ', Avg. power iter. = ' num2str(avgPowerIter)]);
    
    %T = TensorFullM3LDA(Xtilde,Mtilde,lambdaArr,thetaMat,alpha0);
    %T
    if k == 1
        % initialize tensor
        T = TensorFullM3LDA(Xtilde,Mtilde,lambdaArr,thetaMat,alpha0);
    else
        % deflate tensor with the last recovered component
        Temp = zeros(K,K,K);
        for i = 1:K
            Temp(:,:,i) = thetaMat(i,end)*thetaMat(:,end)*thetaMat(:,end)';
        end
        T6 = lambdaArr(end)*Temp;
        T = T - T6;
    end

    % Iterate over L random starts
    lambdaPerStart = zeros(1,L);
    thetaPerStart = zeros(K,L);
    powerIterCounts = zeros(1,L);
    for tau = 1:L
        % Choose theta unifromly over unit sphere
        theta = randn(K,1);
        theta = theta/norm(theta);
        % Power iterations
        diff = zeros(1,NumIter);
        for t = 1:NumIter
            % Compute power iteration update
            thetaNew = zeros(K,1);
            % Compute M3(W,W*theta,W*theta)
            for i = 1:K
                temp = 0;
                for j = 1:K
                    for k = 1:K
                        temp = temp + theta(j)*theta(k)*T(i,j,k);
                    end
                end
                thetaNew(i) = temp;
            end 
            % Update theta
            thetaNew = thetaNew/norm(thetaNew);
            diff(t) = norm(theta-thetaNew);
            theta = thetaNew;
            
            % Check for convergence after 10 iterations
            if t>10
                % Check if maximum error over last 3 iterations is less than
                % threshold
                if max(diff(t-2:t)) < epsilon
                    powerIterCounts(tau) = t;
                    break;
                end
            end
        end % end of power iterations  
        % Compute lambda = M3(W*theta,W*theta,W*theta)
        temp2 = 0;
        for i = 1:K
            for j = 1:K
                for k = 1:K
                    temp2 = temp2 + prod(theta([i,j,k]))*T(i,j,k);
                end
            end
        end
        lambdaPerStart(tau) = temp2;
        thetaPerStart(:,tau) = theta;
    end % end of random start tau
    % Find max component
    maxLambda = max(lambdaPerStart);
    taustar = min(find(lambdaPerStart==maxLambda));
    lambdaArr = [lambdaArr, maxLambda];
    thetaMat = [thetaMat, thetaPerStart(:,taustar)];
    avgPowerIter = mean(powerIterCounts);
end

disp('Tensor power method complete!');

end