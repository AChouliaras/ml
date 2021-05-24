%% SVM email classification 

clear all;

% Change numTrainDocs to choose how many training samples to learn from
% Possibilities: 50, 100, 400, 700
numTrainDocs = 50;

%================================================
% Exercise 1B-1:  Load the data and labels.
[train_matrix, test_matrix, train_labels, test_labels] = load_emails(numTrainDocs);

%================================================
% Exercise 1B-2: Learn and test SVM models.

% Learn svm
% use C = 1 - you can always check whether this affects the 
C = 1;
model = svmtrain(train_matrix,train_labels,'BoxConstraint', C);
predicted_labels = svmclassify(model,test_matrix);

% Display the learned weights
 
%================================================
% Exercise 1B-3: Compute accuracy.
accuracy = mean(double(predicted_labels == test_labels))
acc_all0 = mean(double(0 == test_labels))
sprintf('Accuracy on test set is %f percent',accuracy)

confusionmat(predicted_labels , test_labels)

csvwrite('train_matrix50.txt',train_matrix)
csvwrite('test_matrix50.txt',test_matrix)
csvwrite('train_labels.txt',train_labels)
csvwrite('test_labels.txt',test_labels)