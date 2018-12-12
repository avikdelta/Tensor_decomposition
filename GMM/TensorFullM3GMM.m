function T = TensorFullM3GMM(X,M1,lambdaArr,thetaMat)

%--------------------------------------------------------------
% Author: Avik Ray (avik@utexas.edu)
% Copyright (C) 2015-2018 by Avik Ray
%
% Description: Computes remainder 3rd moment tensor for a GMM 
% after deflation [ref: Anandkumar et al. (2014)].
%
%--------------------------------------------------------------

[d N] = size(X);
T = zeros(d,d,d);
for s = 1:N
    Temp = zeros(d,d,d);
    for i = 1:d
        Temp(:,:,i) = X(i,s)*X(:,s)*X(:,s)';
    end
    T = T + Temp;
end
T = T/N;

T2 = zeros(d,d,d);
for i = 1:d
    %T2(i,:,:) = T2(i,:,:) + M1(i)*eye(d); 
    for l = 1:d
        T2(i,l,l) = T2(i,l,l) + M1(i);
    end
end
for j = 1:d
    %T2(:,j,:) = T2(:,j,:) + M1(j)*eye(d);
    for l = 1:d
        T2(l,j,l) = T2(l,j,l) + M1(j);
    end
end
for k = 1:d
    %T2(:,:,k) = T2(:,:,k) + M1(k)*eye(d);
    for l = 1:d
        T2(l,l,k) = T2(l,l,k) + M1(k);
    end
end

T = T - T2;

% Deflation
M = length(lambdaArr);
if M>0
    T3 = zeros(d,d,d);
    for m = 1:M
        Temp = zeros(d,d,d);
        for i = 1:d
            Temp(:,:,i) = thetaMat(i,m)*thetaMat(:,m)*thetaMat(:,m)';
        end
        T3 = T3 + lambdaArr(m)*Temp;
    end
    T = T - T3;
end

end