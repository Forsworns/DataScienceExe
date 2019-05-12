function K = constructKernel(X,Y,sigma)
%sigma=compute_sigma(X);

K = exp(-sqdist(Y,X)/(2*sigma.*sigma));

function sqd = sqdist(X,Y)

%if exist('w','var') & ~isempty(w)
%  h = sqrt(w(:)'); X = bsxfun(@times,X,h);
%  if eqXY==1 Y = X; else Y = bsxfun(@times,Y,h); end;
%end

% The intervector squared distance is computed as (x-y)Â² = xÂ²+yÂ²-2xy.
% We ensure that no value is negative (which can happen due to precision loss
% when two vectors are very close).
x = sum(X.^2,2);
y = sum(Y.^2,2)';
sqd = max(bsxfun(@plus,x,bsxfun(@plus,y,-2*X*Y')),0);
