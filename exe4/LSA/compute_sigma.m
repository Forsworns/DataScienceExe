function [sigma]= compute_sigma(X)
N=size(X,1);
DIST=zeros(N,N);
    for k=1:size(X,2)
        [a,b]=meshgrid(X(:,k));
        DIST=DIST+(a-b).^2;
    end
    DIST=DIST.^0.5;
    DIST=DIST+diag(ones(N,1)*inf);
    sigma=20*median(min(DIST));