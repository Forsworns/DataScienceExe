% later run concat.m
clear;
clc;
addpath(genpath('..'))
cd('..')
run('vl_setup')

bTest = false;
numLD_path = '.\results\siftLD\numLD.mat';
filesNum_path = '.\results\siftLD\filesNum.mat';
dir_base = '.\data\Animals_with_Attributes2\JPEGImages\';
if ~exist('results','dir')
    mkdir('results')
end
sift_dir = '.\results\sift\';
if ~exist(sift_dir,'dir')
    mkdir(sift_dir)
end
siftLD_dir = '.\results\siftLD\';
if ~exist(siftLD_dir,'dir')
    mkdir(siftLD_dir)
end
%% 遍历目录
tic;
dir_list = dir(dir_base);
dir_num = length(dir_list);
sprintf('开始遍历图片共%d个文件夹',dir_num);
file_nums = zeros(dir_num-2,1);
dir_names = cell(dir_num-2,1);
numLD = [];
for i= 3:dir_num % omit '.' and '..'
    sprintf('第%d组图片%s',i-2,dir_list(i).name);
    
    if dir_list(i).isdir
        dir_names{i-2} = dir_list(i).name; % class name
        dir_path = [dir_base,dir_names{i-2},'\'];
        file_list = dir(dir_path);
        if bTest
            file_nums(i-2) = 10;
        else
            file_nums(i-2) = length(file_list)-2;
        end
        % iterate to view every figure in each class
        for j = 1:file_nums(i-2)
            file_path = [dir_path,file_list(j+2).name];
            image = imread(file_path);
            % imshow(image) % show the figure
           %% compute SIFT
            image = single(rgb2gray(image)); % transform to gray scale images to cal sift local desciptor
            [f,d] = vl_sift(image,'PeakThresh',10); % A frame is a disk of center f(1:2), scale f(3) and orientation f(4)
            % a single sift local descriptor is of size 128X1
            data_dir = [sift_dir,dir_names{i-2}];
            if ~exist(data_dir,'dir')
                mkdir(data_dir)
            end
            data_path = [data_dir,'\',file_list(j+2).name,'.mat'];
            save(data_path,'d')
            numLD = [numLD,size(d,2)];
            %% visualization of the SIFT local descriptor
%             perm = randperm(size(f,2));
%             sel = perm(1:50);
%             h1 = vl_plotframe(f(:,sel));
%             h2 = vl_plotframe(f(:,sel));
%             set(h1,'color','k','linewidth',3);
%             set(h2,'color','y','linewidth',2);
%             h3 = vl_plotsiftdescriptor(d(:,sel),f(:,sel)) ;
%             set(h3,'color','g') ;
        end
    end
end
save(numLD_path,'numLD')
sift_time = toc;




