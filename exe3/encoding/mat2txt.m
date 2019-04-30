%% "APPEND" data to the text files£¡
close all;
clear;
clc;
addpath(genpath('..'))
cd('..')

bow_dir = '.\results\bow\';
vlad_dir = '.\results\vlad\';
fisher_dir = '.\results\fisher\';

if ~exist('.\results\txt','dir')
    mkdir('.\results\txt');
end
bow_txt = '.\results\txt\bow.txt';
vlad_txt = '.\results\txt\vlad.txt';
fisher_txt = '.\results\txt\fisher.txt';

bow_list = dir(bow_dir);
vlad_list = dir(vlad_dir);
fisher_list = dir(fisher_dir);
%% resort the dir(path)
nameCell = cell(length(bow_list)-2,1);
for i = 3:length(bow_list)
    nameCell{i-2} = bow_list(i).name;
end
bow_list = sort_nat(nameCell);

nameCell = cell(length(vlad_list)-2,1);
for i = 3:length(vlad_list)
    nameCell{i-2} = vlad_list(i).name;
end
vlad_list = sort_nat(nameCell);

nameCell = cell(length(fisher_list)-2,1);
for i = 3:length(fisher_list)
    nameCell{i-2} = fisher_list(i).name;
end
fisher_list = sort_nat(nameCell);

bow_len = length(bow_list);
vlad_len = length(vlad_list);
fisher_len = length(fisher_list);

tic;
%% write features to a txt
file = fopen(bow_txt,'a');
for i = 1:bow_len
    encoding_name = [bow_dir,bow_list{i}];
    load(encoding_name,'encoding');
    for j = 1:length(encoding)
        fprintf(file,'%d ',encoding(j));
    end
    fprintf(file,"\r\n");
end
fclose(file);

file = fopen(vlad_txt,'a');
for i = 1:vlad_len
    encoding_name = [vlad_dir,vlad_list{i}];
    load(encoding_name,'encoding');
    for j = 1:length(encoding)
        fprintf(file,'%.4f ',encoding(j));
    end
    fprintf(file,"\r\n");
end
fclose(file);

file = fopen(fisher_txt,'a');
for i = 1:fisher_len
    encoding_name = [fisher_dir,fisher_list{i}];
    load(encoding_name,'encoding');
    for j = 1:length(encoding)
        fprintf(file,'%.4f ',encoding(j));
    end
    fprintf(file,"\r\n");
end
fclose(file);

%% write labels to txt
load('.\results\siftLD\label.mat','labels')
label_txt = '.\results\txt\label.txt';
file = fopen(label_txt,'a');
for j = 1:length(labels)
    fprintf(file,'%d',labels(j));
    fprintf(file,"\r\n");
end
fclose(file);
write_time = toc;