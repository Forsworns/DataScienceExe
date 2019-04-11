% 这里先提取出来local descriptor，然后用matlab进行encoding，和之前提供的特征一样，逐行写入txt中。之后的用法就一样了
addpath("..")

% 遍历目录（下载完后根据目录补

%% compute SIFT
img_name = "";
I = vl_impattern(img_name);
image(I);
I = single(rgb2gray(I)); % transform to gray scale images to cal sift local desciptor
[f,d] = vl_sift(I); % A frame is a disk of center f(1:2), scale f(3) and orientation f(4)
% 更改'PeakThresh'可以修改特征的数量，先抽一张看看时间开销，
%% visualization of the SIFT local descriptor
% perm = randperm(size(f,2));
% sel = perm(1:50);
% h1 = vl_plotframe(f(:,sel));
% h2 = vl_plotframe(f(:,sel));
% set(h1,'color','k','linewidth',3);
% set(h2,'color','y','linewidth',2);
% h3 = vl_plotsiftdescriptor(d(:,sel),f(:,sel)) ;
% set(h3,'color','g') ;


