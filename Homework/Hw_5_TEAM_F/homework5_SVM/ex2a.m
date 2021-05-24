%% RBF SVM example a

clear all;
close all;
clc;

%===================================================
% Exercise 2A-1: Load and visualize data.
% Load and plot data.
[data, labels] = load_twofeature('ex2Data/ex2a.txt');

%===================================================
% Exercise 2A-2: Learn the SVM model.
% Set the gamma parameter
gamma = 100;
% set the C value
C = 1;

% figure
% model = svmtrain(...);

%====================================================
% Exercise 2A-3: RBF Kernel.
% Please, implement the RBF kernel in rbf_kernel.m function.

%====================================================
% Exercise 2A-4: Visualize the decision cost.

% % Plot the image and the contours of scoring function
% step = 0.01;
% [X,Y] = meshgrid(0:step:1,0.4:step:1);
% Z = zeros(size(X));
% 
% for i=1:size(X,1)
%     for j=1:size(X,2)
%         Z(i,j) = rbf_scoring_function([X(i,j),Y(i,j)],model.SupportVectors,model.Alpha,model.Bias,gamma);
%     end
% end
 
% % Plot the data points
% figure
% pos = find(labels == 1);
% neg = find(labels == -1);
% plot(data(pos,1), data(pos,2), 'ko', 'MarkerFaceColor', 'b'); hold on;
% plot(data(neg,1), data(neg,2), 'ko', 'MarkerFaceColor', 'g')

% Superimpose the score function as image

% hold on
% display the decision boundary
% c=contour(X,Y,Z,[0 0],'color','k');
 
% Display other constant cost lines


