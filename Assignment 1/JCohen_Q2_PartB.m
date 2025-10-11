clear all, close all;
N = 10000; p = [0.25, 0.25, 0.25, 0.25];
mu1 = [-1; -1]; Sigma1 = [1, -0.5; -0.5, 1];
mu2 = [1; 1]; Sigma2 = [1, 0.3; 0.3, 1];
mu3 = [-1; 1]; Sigma3 = [1, 0; 0, 1];
mu4 = [1; -1]; Sigma4 = [1, -0.3; -0.3, 1];

%Randomly assign the labels
u = rand(1, N);
true_labels = zeros(1, N);
true_labels(u < p(1)) = 1;
true_labels(u >= p(1) & u < p(1)+p(2)) = 2;
true_labels(u >= p(1)+p(2) & u < p(1)+p(2)+p(3)) = 3;
true_labels(u >= p(1)+p(2)+p(3)) = 4;

% get the total number of samples in each label
N1 = sum(true_labels == 1);
N2 = sum(true_labels == 2);
N3 = sum(true_labels == 3);
N4 = sum(true_labels == 4);

% Generate samples from each class
X = zeros(N, 2);
X(true_labels == 1, :) = mvnrnd(mu1, Sigma1, N1);
X(true_labels == 2, :) = mvnrnd(mu2, Sigma2, N2);
X(true_labels == 3, :) = mvnrnd(mu3, Sigma3, N3);
X(true_labels == 4, :) = mvnrnd(mu4, Sigma4, N4);

likelihoods = zeros(N, 4); %Get the likelihoods for all of the samples
likelihoods(:, 1) = mvnpdf(X, mu1', Sigma1);
likelihoods(:, 2) = mvnpdf(X, mu2', Sigma2);
likelihoods(:, 3) = mvnpdf(X, mu3', Sigma3);
likelihoods(:, 4) = mvnpdf(X, mu4', Sigma4);

% Define the loss matrix Lambda (rows=decision D, cols=true label L)
Lambda = [0, 10, 10, 100;
          1, 0, 10, 100;
          1, 1, 0, 100;
          1, 1, 1, 0];

% Calculate posterior probabilities P(L=j|x) for each sample
posteriors = likelihoods .* repmat(p, N, 1);
posteriors = posteriors ./ repmat(sum(posteriors, 2), 1, 4);

% Calculate expected risk for each decision: R(D=i|x) = sum_j Lambda(i,j) * P(L=j|x)
expected_risks = posteriors * Lambda';

% Select the decision with minimum expected risk (ERM rule)
[~, decided_labels] = min(expected_risks, [], 2);
decided_labels = decided_labels';

% Calculate confusion matrix
confusion_matrix = zeros(4, 4);
for j = 1:4  % (L=j)
    for i = 1:4  % (D=i)
        count_ij = sum((true_labels == j) & (decided_labels == i));
        total_j = sum(true_labels == j);
        confusion_matrix(i, j) = count_ij / total_j;
    end
end

% Calculate empirical average risk using the loss matrix
total_risk = 0;
for j = 1:4
    for i = 1:4
        count_ij = sum((true_labels == j) & (decided_labels == i));
        total_risk = total_risk + Lambda(i, j) * count_ij;
    end
end
average_risk = total_risk / N;

% Calculate empirical probability of error
errors = sum(true_labels ~= decided_labels);
prob_error = errors / N;

% Display results
fprintf('Confusion Matrix P(D=i|L=j) (rows=decided D, cols=true L):\n');
fprintf('        L1      L2      L3      L4\n');
for i = 1:4
    fprintf('D%d: %5.1f%% %5.1f%% %5.1f%% %5.1f%%\n', i, ...
            confusion_matrix(i,1)*100, confusion_matrix(i,2)*100, ...
            confusion_matrix(i,3)*100, confusion_matrix(i,4)*100);
end
fprintf('\n');

figure(1);
set(gcf, 'Position', [100, 100, 800, 700]); % Set figure size
hold on;
axis equal;
grid on;

% Define marker shapes and colors
markers = {'.', 'o', '^', 's'};
marker_size = 4;
colors_correct = [0, 0.7, 0];
colors_incorrect = [0.9, 0, 0];

%Determine what labels are correct and what is note
is_correct = (true_labels == decided_labels);

% Plot each class
for class = 1:4
    class_mask = (true_labels == class);
    
    % Plot correct classifications
    idx_correct = class_mask & is_correct;
    if sum(idx_correct) > 0
        plot(X(idx_correct, 1), X(idx_correct, 2), markers{class}, ...
             'MarkerSize', marker_size, ...
             'MarkerEdgeColor', colors_correct, ...
             'MarkerFaceColor', 'none', ...
             'LineWidth', 0.5);
    end
    
    % Plot incorrect classifications
    idx_incorrect = class_mask & ~is_correct;
    if sum(idx_incorrect) > 0
        plot(X(idx_incorrect, 1), X(idx_incorrect, 2), markers{class}, ...
             'MarkerSize', marker_size, ...
             'MarkerEdgeColor', colors_incorrect, ...
             'MarkerFaceColor', 'none', ...
             'LineWidth', 0.5);
    end
end

xlabel('x_1', 'FontSize', 12);
ylabel('x_2', 'FontSize', 12);
title(sprintf('ERM Classification Results (Average Risk: %.2f) (Error Rate: %.2f%%)\nGreen = Correct, Red = Incorrect', ...
              average_risk, prob_error*100), 'FontSize', 14);

% Legend creation
legend_handles = [];
legend_entries = {};

h1 = plot(NaN, NaN, 'o', 'MarkerEdgeColor', colors_correct, 'MarkerFaceColor', 'none');
h2 = plot(NaN, NaN, 'o', 'MarkerEdgeColor', colors_incorrect, 'MarkerFaceColor', 'none');
legend_handles = [h1, h2];
legend_entries = {'Correct', 'Incorrect'};

% Add marker shape legend
for class = 1:4
    h = plot(NaN, NaN, markers{class}, ...
             'MarkerSize', marker_size+2, ...
             'MarkerEdgeColor', 'k', ...
             'MarkerFaceColor', 'none', ...
             'LineWidth', 1);
    legend_handles(end+1) = h;
    legend_entries{end+1} = sprintf('Class %d', class);
end

legend(legend_handles, legend_entries, 'Location', 'best', 'FontSize', 10);

hold off;
