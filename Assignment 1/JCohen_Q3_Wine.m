clear all, close all;
data_dir = 'wine+quality';
alpha = 0.0001; % Regularization parameter

% Load wine csvs and combine them into one dataset
filename_red = fullfile(data_dir, 'winequality-red.csv');
data_red = readtable(filename_red);
filename_white = fullfile(data_dir, 'winequality-white.csv');
data_white = readtable(filename_white);
data = [data_red; data_white];
X = table2array(data(:, 1:end-1)); % Get all columns except quality
true_labels = data.quality;

% Get dimensionsa and class information
[N, d] = size(X);
classes = unique(true_labels);
n_classes = length(classes);

fprintf('Dataset: %d samples, %d features, %d classes\n', N, d, n_classes);
fprintf('Classes: %s\n', mat2str(classes'));

mu = zeros(n_classes, d);
Sigma = cell(n_classes, 1);
p = zeros(1, n_classes); % Class priors

for i = 1:n_classes
    class_idx = (true_labels == classes(i));
    X_class = X(class_idx, :);
    n_class = sum(class_idx);
    
    % Estimate mean
    mu(i, :) = mean(X_class, 1);
    
    % Estimate covariance with regularization
    C_sample = cov(X_class);
    trace_C = trace(C_sample);
    C_regularized = C_sample + (alpha * trace_C / d) * eye(d);
    Sigma{i} = C_regularized;
    
    % Estimate prior
    p(i) = n_class / N;
    
    fprintf('Class %d: %d samples (%.1f%%)\n', classes(i), n_class, p(i)*100);
end

%Compute the likelihoods
likelihoods = zeros(N, n_classes);
for i = 1:n_classes
    likelihoods(:, i) = mvnpdf(X, mu(i, :), Sigma{i});
end

% Decide the class with maximum likelihood (weighted by prior)
[~, decided_idx] = max(likelihoods .* p, [], 2);
decided_labels = classes(decided_idx);

% Calculate confusion matrix
confusion_matrix = zeros(n_classes, n_classes);
for j = 1:n_classes  %L
    for i = 1:n_classes  %D
        count_ij = sum((true_labels == classes(j)) & (decided_labels == classes(i)));
        total_j = sum(true_labels == classes(j));
        confusion_matrix(i, j) = count_ij / total_j;
    end
end

% Calculate empirical probability of error
errors = sum(true_labels ~= decided_labels);
prob_error = errors / N;

% Display results
fprintf('\nConfusion Matrix P(D=i|L=j) (rows=decided D, cols=true L):\n');
fprintf('D\\L  |');
for j = 1:n_classes
    fprintf('  %4d  |', classes(j));
end
fprintf('\n');
for i = 1:n_classes
    fprintf(' %2d  |', classes(i));
    for j = 1:n_classes
        fprintf(' %5.1f%% |', confusion_matrix(i,j)*100);
    end
    fprintf('\n');
end
fprintf('\nError Rate: %.2f%%\n', prob_error*100);

% Visualization using PCA

% Perform PCA
[coeff, score, ~, ~, explained] = pca(X);
X_2d = score(:, 1:2);

figure(1);
set(gcf, 'Position', [100, 100, 800, 700]);
hold on;
axis equal;
grid on;

markers = {'.', 'o', '^', 's', 'd', 'v', '>', '<'};
marker_size = 6;
colors_correct = [0, 0.7, 0];
colors_incorrect = [0.9, 0, 0];

% Find the labels that are correct and not correct
is_correct = (true_labels == decided_labels);

% Plot each class
for class = 1:n_classes
    class_mask = (true_labels == classes(class));
    marker_idx = mod(class-1, length(markers)) + 1;
    
    % Plot correct classifications (green)
    idx_correct = class_mask & is_correct;
    if sum(idx_correct) > 0
        plot(X_2d(idx_correct, 1), X_2d(idx_correct, 2), markers{marker_idx}, ...
             'MarkerSize', marker_size, ...
             'MarkerEdgeColor', colors_correct, ...
             'MarkerFaceColor', 'none', ...
             'LineWidth', 0.5);
    end
    
    % Plot incorrect classifications (red)
    idx_incorrect = class_mask & ~is_correct;
    if sum(idx_incorrect) > 0
        plot(X_2d(idx_incorrect, 1), X_2d(idx_incorrect, 2), markers{marker_idx}, ...
             'MarkerSize', marker_size, ...
             'MarkerEdgeColor', colors_incorrect, ...
             'MarkerFaceColor', 'none', ...
             'LineWidth', 0.5);
    end
end

xlabel(sprintf('PC1 (%.1f%% var)', explained(1)), 'FontSize', 12);
ylabel(sprintf('PC2 (%.1f%% var)', explained(2)), 'FontSize', 12);
title(sprintf('MAP Classification Results (Error Rate: %.2f%%)\nGreen = Correct, Red = Incorrect', ...
              prob_error*100), 'FontSize', 14);

% Legend creation
legend_handles = [];
legend_entries = {};

h1 = plot(NaN, NaN, 'o', 'MarkerEdgeColor', colors_correct, 'MarkerFaceColor', 'none');
h2 = plot(NaN, NaN, 'o', 'MarkerEdgeColor', colors_incorrect, 'MarkerFaceColor', 'none');
legend_handles = [h1, h2];
legend_entries = {'Correct', 'Incorrect'};

% Add class legend
for class = 1:n_classes
    marker_idx = mod(class-1, length(markers)) + 1;
    h = plot(NaN, NaN, markers{marker_idx}, ...
             'MarkerSize', marker_size+2, ...
             'MarkerEdgeColor', 'k', ...
             'MarkerFaceColor', 'none', ...
             'LineWidth', 1);
    legend_handles(end+1) = h;
    legend_entries{end+1} = sprintf('Class %d', classes(class));
end

legend(legend_handles, legend_entries, 'Location', 'best', 'FontSize', 10);

hold off;

% 3D visualization
figure(2);
set(gcf, 'Position', [150, 150, 800, 700]);
X_3d = score(:, 1:3);

hold on;
grid on;
view(3);

marker_size_3d = 12;

% Plot each class
for class = 1:n_classes
    class_mask = (true_labels == classes(class));
    marker_idx = mod(class-1, length(markers)) + 1;
    
    % Plot correct classifications
    idx_correct = class_mask & is_correct;
    if sum(idx_correct) > 0
        scatter3(X_3d(idx_correct, 1), X_3d(idx_correct, 2), X_3d(idx_correct, 3), ...
                 marker_size_3d, markers{marker_idx}, ...
                 'MarkerEdgeColor', colors_correct, 'LineWidth', 1);
    end
    
    % Plot incorrect classifications
    idx_incorrect = class_mask & ~is_correct;
    if sum(idx_incorrect) > 0
        scatter3(X_3d(idx_incorrect, 1), X_3d(idx_incorrect, 2), X_3d(idx_incorrect, 3), ...
                 marker_size_3d, markers{marker_idx}, ...
                 'MarkerEdgeColor', colors_incorrect, 'LineWidth', 1);
    end
end

xlabel(sprintf('PC1 (%.1f%%)', explained(1)), 'FontSize', 12);
ylabel(sprintf('PC2 (%.1f%%)', explained(2)), 'FontSize', 12);
zlabel(sprintf('PC3 (%.1f%%)', explained(3)), 'FontSize', 12);
title(sprintf('3D MAP Classification (Error Rate: %.2f%%)', prob_error*100), 'FontSize', 14);

% Legend for 3D plot
legend_handles_3d = [];
legend_entries_3d = {};

h1_3d = scatter3(NaN, NaN, NaN, marker_size_3d, 'o', 'MarkerEdgeColor', colors_correct);
h2_3d = scatter3(NaN, NaN, NaN, marker_size_3d, 'o', 'MarkerEdgeColor', colors_incorrect);
legend_handles_3d = [h1_3d, h2_3d];
legend_entries_3d = {'Correct', 'Incorrect'};

for class = 1:n_classes
    marker_idx = mod(class-1, length(markers)) + 1;
    h = scatter3(NaN, NaN, NaN, marker_size_3d, markers{marker_idx}, ...
                 'MarkerEdgeColor', 'k', 'LineWidth', 1);
    legend_handles_3d(end+1) = h;
    legend_entries_3d{end+1} = sprintf('Class %d', classes(class));
end

legend(legend_handles_3d, legend_entries_3d, 'Location', 'best', 'FontSize', 10);

hold off;