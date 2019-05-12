% Rahaf Aljundi 2014
% Choose the best points to be landmarks depending on multiple percentiles of sigma
% Source_Data: source data (samples as rows, features as columns) 
% Target_Data: target data.
%src,tgt: the name of the datasets used for saving the landmarks file.
%landmarks: the selected landmarks.

function landmarks =choose_landmarks(Source_Data,Target_Data,src,tgt)


%initialization

all_data_points=[Target_Data;Source_Data];

N=size(all_data_points,1);
Inds=zeros(N,1);
dis_vals=zeros(N,1);

%loop around the percentiles of sigma
for i=1: 100


sigma=compute_sigma_point_percentile(Source_Data,Target_Data,i);
%vote for landmarks from source and target
[Inds,dis_vals]= choose_best_points_per_sigma(Source_Data,Target_Data,all_data_points,sigma,Inds,dis_vals);
i
end

%get the points that has one vote at least
big_inds=find(Inds>1);
Inds=all_data_points(big_inds,:);
dis_vals=dis_vals(big_inds,:);
%sort them 
[S,S_Inds]=sort(dis_vals,'descend');
 Inds=Inds(S_Inds,:);
 dis_vals=dis_vals(S_Inds,:);
 %get the landmarks
landmarks=Inds;
 name=char(strcat('landmarks1_', src,  '_', tgt, '.mat'));
save(name,'Inds');
end
