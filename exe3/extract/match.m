clear;
clc;
addpath(genpath('..'))
run('vl_setup')
I1 = imread('hamster_10060.jpg') ;
I1 = single(rgb2gray(I1));

I2 = imread('hamster_10072.jpg') ;
I2 = single(rgb2gray(I2));

I3 = imread('tiger_10018.jpg') ;
I3 = single(rgb2gray(I3));

[f1, d1] = vl_sift(I1,'PeakThresh',10) ;
[f2, d2] = vl_sift(I2,'PeakThresh',10) ;
[f3, d3] = vl_sift(I3,'PeakThresh',10) ;
[matches1, scores1] = vl_ubcmatch(d1, d2) ;
[matches2, scores2] = vl_ubcmatch(d1, d3) ;

figure(1)
colormap(gray);
imagesc(I1);
for i=1:length(matches1)
    h1 = vl_plotsiftdescriptor(d1(:,matches1(1,:)),f1(:,matches1(1,:))) ;
    set(h1,'color','b') ;      
end 

figure(2)
colormap(gray);
imagesc(I2);
for i=1:length(matches1)
    h2 = vl_plotsiftdescriptor(d2(:,matches1(2,:)),f2(:,matches1(2,:))) ;
    set(h2,'color','g') ;       
end 

figure(3)
colormap(gray);
imagesc(I1);
for i=1:length(matches2)
    h3 = vl_plotsiftdescriptor(d1(:,matches2(1,:)),f1(:,matches2(1,:))) ;
    set(h3,'color','b') ;
end

figure(4)
colormap(gray);
imagesc(I3);
for i=1:length(matches2)
    h4 = vl_plotsiftdescriptor(d3(:,matches2(2,:)),f3(:,matches2(2,:))) ;
    set(h4,'color','g') ;
end
