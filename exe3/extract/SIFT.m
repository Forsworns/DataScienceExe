% 这里先提取出来local descriptor，然后用matlab进行encoding，和之前提供的特征一样，逐行写入txt中。之后的用法就一样了
addpath('..')
run('vl_setup')

sift_dir = '.\results\sift\';
% 遍历目录
dir_base = '.\data\Animals_with_Attributes2\JPEGImages\';
dir_list = dir(dir_base);
dir_num = length(file_list);
sprintf('开始遍历共%d个文件',dir_num);

file_nums = zeros(dir_num-2,1);
dir_names = cell(dir_num-2,1);
for i= 1:dir_num % omit '.' and '..'
    if strcmp(dir_list(i).name,'.')==1 || strcmp(dir_list(i).name,'..')==1
        continue
    end
    if dir_list(i).isdir
        dir_names{i} = dir_list(i).name; % class name
        dir_path = [dir_base,dir_names{i},'\'];
        file_list = dir(dir_path);
        file_nums(i) = length(dir_path);
        % iterate to view every figure in each class
        for j = 1:file_nums(i)
            if strcmp(file_list(j).name,'.')==1 || strcmp(file_list(j).name,'..')==1
                continue
            end
            file_path = [dir_path,file_list(j).name];
            image = imread(file_path);
            imshow(image) % show the figure
           %% compute SIFT
            image = single(rgb2gray(image)); % transform to gray scale images to cal sift local desciptor
            [f,d] = vl_sift(image); % A frame is a disk of center f(1:2), scale f(3) and orientation f(4)
            data_dir = [sift_dir,dir_names{i}];
            if ~exist(data_dir,'dir')
                mkdir(data_dir)
            end
            data_path = [data_dir,'\',file_list(j).name,'.mat'];
            save(data_path,'f','d')
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




