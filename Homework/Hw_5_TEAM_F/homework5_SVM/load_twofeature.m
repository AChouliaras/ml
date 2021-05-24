function [ data, labels ] = load_twofeature(filename)

fileID = fopen(filename);

formatSpec = '%d %*d:%f %*d:%f';
C = textscan(fileID,formatSpec,'Delimiter','\n','CollectOutput', true);

labels = C{1};
data = C{2};

% Plot the data points
figure
pos = find(labels == 1);
neg = find(labels == -1);
plot(data(pos,1), data(pos,2), 'ko', 'MarkerFaceColor', 'b'); hold on;
plot(data(neg,1), data(neg,2), 'ko', 'MarkerFaceColor', 'g')


end

