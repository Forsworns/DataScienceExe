% Rahaf Aljundi 2014
% gives the sigma that remebles the percentile of the distances between the two matrices X and Y.
% X and Y shoud have the same number of features.
function [sigma]=compute_sigma_point_percentile(X,Y,percentile)
	DIST=cvEucdist(X',Y');
    DIST=DIST.^0.5;
    sigma=prctile(prctile(DIST,percentile),percentile);

