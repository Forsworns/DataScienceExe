%Rahaf Aljundi 2014
%cl_source_points: source points to select the landmarks from.
%cl_target_points: target points to select the landmarks from.
%sigma: the current sigma scale.
%Inds: accumelating voting vector{initial state empty to be filled by the algorithm}.
%dis_vals: accumelating distance vector{initial state empty to be filled by the algorithm}.



function [Inds,dis_vals]=choose_best_points_per_sigma(cl_source_points,cl_target_points,all_points,sigma,Inds,dis_vals)
if(~exist('all_points','var'))
    all_points=[cl_source_points;cl_target_points];
end
N=size(all_points,1);

sim_th=0.3;

%loop around all the points and check if they worth to be choosen
for i=1:N
    sample=all_points(i,:);
   
    %calc the differences of the source points
    src_kernel=constructKernel(sample,cl_source_points,sigma);
    src_std=std(src_kernel);
    src_mean=mean(src_kernel);
    %calc the differences of the target points
    tgt_kernel=constructKernel(sample,cl_target_points,sigma);
    tgt_std=std(tgt_kernel);
    tgt_mean=mean(tgt_kernel);
    std_val=src_std*src_std+tgt_std*tgt_std;
    src_tgt_sim=normpdf(src_mean, tgt_mean, std_val);
   %base norm
     max_sim=normpdf(0, 0, std_val);
    act_sim=src_tgt_sim/max_sim;
  if( act_sim>1)
      act_sim
  end
     if act_sim>sim_th
        Inds(i,1)=Inds(i,1)+1;
        dis_vals(i,1)=dis_vals(i,1)+act_sim;
     end
    
end

    
end
