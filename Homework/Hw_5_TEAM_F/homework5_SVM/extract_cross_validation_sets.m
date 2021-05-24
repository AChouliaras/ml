clear all;
clc;

fileID = fopen('ex2Data/ex2b.txt');

formatSpec = '%d %*d:%f %*d:%f';
C = textscan(fileID,formatSpec,'Delimiter','\n','CollectOutput', true);

labels = C{1};
data = C{2};

rng(0)

R = randperm(211); %211 data points

%Re-organize data according to random permutation
data = data(R,:);
labels = labels(R);

%Build 3 sets for cross-validation

n = 70;

V1 = data(1:n,:);
labelsV1 = labels(1:n,:);

V2 = data(n+1:2*n,:);
labelsV2 = labels(n+1:2*n,:);

V3 = data(2*n+1:end,:);
labelsV3 = labels(2*n+1:end,:);

V = {};
V{1}=V1;
V{2}=V2;
V{3}=V3;

L = {};
L{1}=labelsV1;
L{2}=labelsV2;
L{3}=labelsV3;

save ex2Data/V.mat V
save ex2Data/L.mat L
