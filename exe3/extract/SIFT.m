% ��������ȡ����local descriptor��Ȼ����matlab����encoding����֮ǰ�ṩ������һ��������д��txt�С�֮����÷���һ����
addpath("..")

% ����Ŀ¼������������Ŀ¼��

%% compute SIFT
img_name = "";
I = vl_impattern(img_name);
image(I);
I = single(rgb2gray(I)); % transform to gray scale images to cal sift local desciptor
[f,d] = vl_sift(I); % A frame is a disk of center f(1:2), scale f(3) and orientation f(4)
% ����'PeakThresh'�����޸��������������ȳ�һ�ſ���ʱ�俪����
%% visualization of the SIFT local descriptor
% perm = randperm(size(f,2));
% sel = perm(1:50);
% h1 = vl_plotframe(f(:,sel));
% h2 = vl_plotframe(f(:,sel));
% set(h1,'color','k','linewidth',3);
% set(h2,'color','y','linewidth',2);
% h3 = vl_plotsiftdescriptor(d(:,sel),f(:,sel)) ;
% set(h3,'color','g') ;


