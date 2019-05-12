%Rahaf Aljundi 2014
%src the source data set
%tgt the target data set
%sampledNum the number of samples selected for the kernel construction
%Subspace_Dim1 the reduced supspace dimension
%Subspace_Dim2 the reduced supspace dimension
%Dimensions used previously
%Webcam dslr   (37 37)
%Webcam Caltech  (11 12)
%caltech dslr (11 10)
%caltech webcam (11 15)
%caltech amazon (20 17)
%amazon webcam (15 9)
%Amazon dslr  (10 12)
%Amazon caltech  (13 10)
%-------------------------I. setup source/target domains adaptation----------------------
 function Total_accuracy=LSA(Source_Data,Target_Data,Source_label,Target_label,landmarks,sigma,src_Dim,tgt_Dim)

Source_Data=NormalizeData(Source_Data);
Target_Data=NormalizeData(Target_Data);
% project the source and target on a shared space using a kernel w.r.t the landmarks.
source_kernel=constructKernel(landmarks,Source_Data,sigma);

target_kernel=constructKernel(landmarks,Target_Data,sigma);



source_kernel = NormalizeData(source_kernel);

target_kernel = NormalizeData(target_kernel);
%% Alignment Step %%
% PCA
[Xs,D,ES] = princomp(source_kernel);
[Xt,D,ET] = princomp(target_kernel);

% create subspace
% dimension reduction
Xs = Xs(:,1:src_Dim);
Xt = Xt(:,1:tgt_Dim);

% Subspace alignment and projections
Target_Aligned_Source_Data = source_kernel*(Xs * Xs'*Xt);

Target_Projected_Data = target_kernel*Xt;
%% Having the new representation (Target_Aligned_Source_Data and Target_Projected_Data), you can use any classifier. 
%% SVM CLASSIFIER %%
% svm options ( extract the best options by cross validation)
svmopts=['-c 2 -h 0 -t 0 -g 1'];

% train SVM
Xt=Target_Aligned_Source_Data;
Y=Source_label;
model=libsvmtrain(Y, Xt, svmopts);
% test SVM on test data
tXt=Target_Projected_Data;
tY=Target_label;
[tYout, Total_accuracy, tYext]=libsvmpredict(tY,tXt,model,'');
r=find(tYout==Target_label);
Total_accuracy = length(r)/length(Target_label)*100; 


fprintf('MinAcc MINMIN %1.2f  \n',Total_accuracy);
end
