%% SVM 2D feature classification 

clear all;
close all;
clc;

%=================================
% Exercise 1A-1: Load and plot data
[data, labels] = load_twofeature('ex1Data/twofeature.txt');

%=================================
% Exercise 1A-2: Learn and visualize model

% Set the cost
C = 1;

% Learn the model
figure;
model = svmtrain(data,labels,'ShowPlot',true,'BoxConstraint', C);

% Retrieve the different elements from the model
SVs = model.SupportVectors; %support vectors
sv_coef = model.Alpha; %weights
b = model.Bias; %bias

%=================================
% Exercise 1A-3: Liner Kernel function.
% Please implement the linear kernel function at linear_kernel.m function.

%=================================
% Exercise 1A-4: Plot the decision boundary

% Compute the weights 

% Display the decision boundary 
 
% %highlight support vectors
% plot(SVs(:,1),SVs(:,2),'r*')
% title(sprintf('SVM Linear Classifier with C = %g', C), 'FontSize', 14)
