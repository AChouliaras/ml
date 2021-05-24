function [train_matrix, test_matrix, train_labels, test_labels] = load_emails(numTrainDocs)

numTestDocs = 260;
numTokens = 2500;

%Load training data
M = dlmread(sprintf('ex1Data/emails/train-features-%d.txt',numTrainDocs), ' ');
spmatrix = sparse(M(:,1), M(:,2), M(:,3), numTrainDocs, numTokens);
train_matrix = full(spmatrix);

train_labels = dlmread(sprintf('ex1Data/emails/train-labels-%d.txt',numTrainDocs));

%Load test data
M = dlmread('ex1Data/emails/test-features.txt', ' ');
spmatrix = sparse(M(:,1), M(:,2), M(:,3), numTestDocs, numTokens);
test_matrix = full(spmatrix);

test_labels = dlmread('ex1Data/emails/test-labels.txt');

end

