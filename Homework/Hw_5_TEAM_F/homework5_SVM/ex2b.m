%% RBF SVM example b

clear all;
clc;

%========================================================
% Exercise 2B-1: Load and visualize data.
% Load and plot data
[data, labels] = load_twofeature('ex2Data/ex2b.txt');

%========================================================
% Exercise 2B-2: Learn SVM models for different hyperparameter values.
% Set the parameters
C = [1 1000];
gamma = [1 10 100 1000];

for i=1:length(C)
    for j=1:length(gamma)

        figure
        model = svmtrain(data,labels,'kernel_function','rbf','BoxConstraint', C(i),'rbf_sigma',1/sqrt(2*gamma(j)),'ShowPlot',true);
        title(sprintf('C=%g, \\gamma = %g', C(i),gamma(j)), 'FontSize', 14);
        
    end
end
