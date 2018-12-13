function T = TensorFullM3LDA(X,M,lambdaArr,thetaMat,alpha0)

%--------------------------------------------------------------
% Author: Avik Ray (avik@utexas.edu)
% Copyright (C) 2015-2018 by Avik Ray
%
% Description: Computes remainder 3rd moment tensor for a LDA 
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

% T2
T2 = zeros(d,d,d);
for s = 1:N
    Temp = zeros(d,d,d);
    for i = 1:d
        Temp(:,:,i) = X(i,s)*X(:,s)*M';
    end
    T2 = T2 + Temp;
end
T2 = T2/N;

% T3
T3 = zeros(d,d,d);
for s = 1:N
    Temp = zeros(d,d,d);
    for i = 1:d
        Temp(:,:,i) = X(i,s)*M*X(:,s)';
    end
    T3 = T3 + Temp;
end
T3 = T3/N;

% T4
T4 = zeros(d,d,d);
for s = 1:N
    Temp = zeros(d,d,d);
    for i = 1:d
        Temp(:,:,i) = M(i)*X(:,s)*X(:,s)';
    end
    T4 = T4 + Temp;
end
T4 = T4/N;

% T5
T5 = zeros(d,d,d);
for i = 1:d
    T5(:,:,i) = M(i)*M*M';
end

% Final
T = T - (alpha0/(alpha0+2))*(T2+T3+T4) + (2*alpha0^2/((alpha0+2)*(alpha0+1)))*T5;

% Deflation
M = length(lambdaArr);
if M>0
    T6 = zeros(d,d,d);
    for m = 1:M
        Temp = zeros(d,d,d);
        for i = 1:d
            Temp(:,:,i) = thetaMat(i,m)*thetaMat(:,m)*thetaMat(:,m)';
        end
        T6 = T6 + lambdaArr(m)*Temp;
    end
    T = T - T6;
end

end