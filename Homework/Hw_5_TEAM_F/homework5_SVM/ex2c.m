%% Cross-validation: determine best set of parameters
clear all;
close all;

% ================================================
% Exercise 2C-1: Load data.
% The data points have been split into 3 sets V1, V2, V3 for cross-validation.
% Load these and their corresponding labels here:
load ex2Data/V.mat %V is a cell, Vi is contained in V{i} 
load ex2Data/L.mat %L is a cell, labelsVi are contained in L{i}

% ================================================
% Exercise 2C-2: Perform cross-validation.
% Create a meshgrid of (C, gamma) parameters
[X,Y] = meshgrid(logspace(-1,3,5),logspace(-1,3,5));

A = zeros(size(X)); %will contain average accuracy for each parameter combination (C,gamma)

n = 3; %number of sets

for i=1:size(X,1)
    for j=1:size(X,2)
        C = X(i,j);
        gamma = Y(i,j);
        sigma = sqrt(0.5/gamma);
        acc = zeros(1,n); %will contain accuracy for each set (accuracy of svm learned on union of all the other sets), ...
        %for one given combination of parameters (C,gamma) 
        for v=1:n %cross-validation set index
            test_data = V{v};
            test_labels = L{v};
            %extract data from other sets
            train_data = [];
            train_labels = [];
            for otherset=1:n
                if otherset ~= v
                    train_data = [train_data; V{otherset}];
                    train_labels = [train_labels; L{otherset}];
                end
            end
            %train model
            model = svmtrain(train_data,train_labels,'kernel_function','rbf','BoxConstraint',C,'rbf_sigma',sigma);
            %test model
            predicted_labels = svmclassify(model,test_data);
            %compute accuracy (compare predicted_labels to test_labels)
            acc(v) = mean(double(predicted_labels == test_labels));
        end
        avg_acc = mean(acc);
        A(i,j) = avg_acc;
    end
end

%Best combination of parameters

[value, location] = max(A(:));
[r,c] = ind2sub(size(A),location); 

sprintf('Best parameters are C=%d and gamma=%d, with a score of %f',X(r,c),Y(r,c),A(r,c))