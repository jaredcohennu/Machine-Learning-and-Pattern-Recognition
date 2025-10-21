clear all, close all,

% Set all of the parameters specified in the problem
N_validate = 10000; p0 = 0.6; p1 = 0.4;
w01 = 0.5; w02 = 0.5;
mu01 = [-0.9; -1.1]; mu02 = [0.8; 0.75];
w11 = 0.5; w12 = 0.5;
mu11 = [-1.1; 0.9]; mu12 = [0.9; -0.75];
C = [0.75, 0; 0, 1.25];
Sigma01 = C; Sigma02 = C; Sigma11 = C; Sigma12 = C;

% Dataset sizes for training
N_train_sizes = [50, 500, 5000];
num_datasets = length(N_train_sizes);

%Generate the validation set
u_validate = rand(1, N_validate) >= p0;
N0_validate = sum(u_validate == 0);
N1_validate = sum(u_validate == 1);

n01_validate = round(N0_validate * w01);
n02_validate = N0_validate - n01_validate;
r01_validate = mvnrnd(mu01, Sigma01, n01_validate);
r02_validate = mvnrnd(mu02, Sigma02, n02_validate);
r0_validate = [r01_validate; r02_validate];

n11_validate = round(N1_validate * w11);
n12_validate = N1_validate - n11_validate;
r11_validate = mvnrnd(mu11, Sigma11, n11_validate);
r12_validate = mvnrnd(mu12, Sigma12, n12_validate);
r1_validate = [r11_validate; r12_validate];

x_validate = zeros(2, N_validate);
label_validate = zeros(1, N_validate);

x_validate(:, u_validate == 0) = r0_validate';
label_validate(u_validate == 0) = 0;
x_validate(:, u_validate == 1) = r1_validate';
label_validate(u_validate == 1) = 1;

Nc_validate = [N0_validate, N1_validate];

results = struct(); %Create struct to help store results

%Iterate through the datasets to apply to all of the classifiers
for k = 1:num_datasets
    N_train = N_train_sizes(k);
    
    fprintf('Training set: N=%d\n', N_train);
    
    % Generate training set
    u_train = rand(1, N_train) >= p0;
    N0_train = sum(u_train == 0);
    N1_train = sum(u_train == 1);
    
    n01_train = round(N0_train * w01);
    n02_train = N0_train - n01_train;
    r01_train = mvnrnd(mu01, Sigma01, n01_train);
    r02_train = mvnrnd(mu02, Sigma02, n02_train);
    r0_train = [r01_train; r02_train];
    
    n11_train = round(N1_train * w11);
    n12_train = N1_train - n11_train;
    r11_train = mvnrnd(mu11, Sigma11, n11_train);
    r12_train = mvnrnd(mu12, Sigma12, n12_train);
    r1_train = [r11_train; r12_train];
    
    % Combine training data
    X_train = [r0_train; r1_train];
    y_train = [zeros(N0_train, 1); ones(N1_train, 1)];
    
    % Add bias term
    X_train_aug = [X_train, ones(N_train, 1)];
    
    % Initial parameters
    params_init = zeros(3, 1);
    
    % Define negative log-likelihood function so that we can reuse it
    objectiveFunc = @(params) negLogLikelihood(params, X_train_aug, y_train);
    
    % Use fminsearch as suggested in the problem
    options = optimset('MaxIter', 10000, 'TolFun', 1e-6, 'Display', 'off');
    params_opt = fminsearch(objectiveFunc, params_init, options);
    
    %Get the optimal weights and bias
    w_opt = params_opt(1:2);
    b_opt = params_opt(3);
    
    % Compute logistic posterior on training set (for reference)
    p_L1_train = sigmoid(X_train_aug * params_opt);
    
    % Augment validation features with bias term and compute posterior
    X_validate_aug = [x_validate', ones(N_validate, 1)];
    p_L1_validate = sigmoid(X_validate_aug * params_opt);
    
    % Use posterior probability as discriminant score
    disc_est = p_L1_validate;
    results(k).disc_est = disc_est;
    
    % Sweep to find the optimal threshold
    gammaRange = linspace(0, 1, 1000);  % Threshold on posterior probability [0,1]
    P_error_sweep = zeros(1, length(gammaRange));
    
    for i = 1:length(gammaRange)
        decision_sweep = (disc_est >= gammaRange(i));
        p10_sweep = sum(decision_sweep == 1 & label_validate' == 0) / Nc_validate(1);
        p01_sweep = sum(decision_sweep == 0 & label_validate' == 1) / Nc_validate(2);
        P_error_sweep(i) = p10_sweep * p0 + p01_sweep * p1;
    end
    
    % Find the minimum P(error) threshold
    [P_error_min, idx_min] = min(P_error_sweep);
    threshold_opt = gammaRange(idx_min);
    
    % Compute confusion matrix at optimal threshold
    decision_opt = (disc_est >= threshold_opt);
    p10_opt = sum(decision_opt == 1 & label_validate' == 0) / Nc_validate(1);
    p01_opt = sum(decision_opt == 0 & label_validate' == 1) / Nc_validate(2);
    p11_opt = sum(decision_opt == 1 & label_validate' == 1) / Nc_validate(2);
    p00_opt = sum(decision_opt == 0 & label_validate' == 0) / Nc_validate(1);
    
    results(k).N_train = N_train;
    results(k).params_opt = params_opt;
    results(k).threshold_opt = threshold_opt;
    results(k).p00 = p00_opt;
    results(k).p10 = p10_opt;
    results(k).p01 = p01_opt;
    results(k).p11 = p11_opt;
    results(k).P_error = P_error_min;
end

%Generate the ROC curve
figure(1), clf, hold on,

colors_rgb = {[0, 0, 1], [0, 0.5, 0], [1, 0, 1]};
thresholdRange_roc = linspace(0, 1, 500);

% Estimate and plot ROC curve for each classifier
for k = 1:num_datasets
    disc_k = results(k).disc_est;
    
    P_TP_roc = zeros(1, length(thresholdRange_roc));
    P_FP_roc = zeros(1, length(thresholdRange_roc));
    
    for i = 1:length(thresholdRange_roc)
        decision = (disc_k >= thresholdRange_roc(i));
        P_FP_roc(i) = sum(decision == 1 & label_validate' == 0) / Nc_validate(1);
        P_TP_roc(i) = sum(decision == 1 & label_validate' == 1) / Nc_validate(2);
    end
    
    % Plot ROC curve
    plot(P_FP_roc, P_TP_roc, 'LineWidth', 2.5, 'Color', colors_rgb{k}, ...
        'DisplayName', sprintf('N=%d', N_train_sizes(k)));
    
    % Mark min-P(error)
    plot(results(k).p10, results(k).p11, 'o', 'Color', colors_rgb{k}, ...
        'MarkerSize', 12, 'LineWidth', 2, 'MarkerFaceColor', colors_rgb{k}, ...
        'DisplayName', sprintf('N=%d: τ=%.2f, P(err)=%.4f', ...
        N_train_sizes(k), results(k).threshold_opt, results(k).P_error));
end

grid on,
xlabel('False Positive Rate: P(D=1|L=0)', 'FontSize', 12);
ylabel('True Positive Rate: P(D=1|L=1)', 'FontSize', 12);
title('ROC Curves: Logistic Linear Function-Based Classifiers', 'FontSize', 14);
legend('show', 'Location', 'southeast', 'FontSize', 9);
axis([0 1 0 1]);
set(gca, 'FontSize', 11);

%Decision boundary plot

% Create a grid for visualization
x1_range = linspace(-5, 5, 300);
x2_range = linspace(-5, 5, 300);
[X1_grid, X2_grid] = meshgrid(x1_range, x2_range);

for k = 1:num_datasets
    figure(k+1), clf, hold on,
    
    % Plot validation data
    idx_class0 = label_validate == 0;
    idx_class1 = label_validate == 1;
    plot(x_validate(1, idx_class0), x_validate(2, idx_class0), 'b.', ...
        'MarkerSize', 6, 'DisplayName', 'Class 0 (validation)');
    plot(x_validate(1, idx_class1), x_validate(2, idx_class1), 'r.', ...
        'MarkerSize', 6, 'DisplayName', 'Class 1 (validation)');
    
    % Compute decision boundary
    X_grid_linear = [X1_grid(:), X2_grid(:), ones(numel(X1_grid), 1)];
    
    Z = sigmoid(X_grid_linear * results(k).params_opt);
    Z_grid = reshape(Z, size(X1_grid));
    
    % Plot decision boundary at optimal threshold
    contour(X1_grid, X2_grid, Z_grid, [results(k).threshold_opt, results(k).threshold_opt], ...
       'k-', 'LineWidth', 3, 'DisplayName', sprintf('Decision Boundary (τ=%.2f)', results(k).threshold_opt));
    
    contour(X1_grid, X2_grid, Z_grid, [0.5, 0.5], ...
        'g--', 'LineWidth', 2, 'DisplayName', 'P(L=1|x)=0.5');
    
    grid on,
    xlabel('x_1', 'FontSize', 12);
    ylabel('x_2', 'FontSize', 12);
    title(sprintf('Logistic-Linear Classifier: N=%d, P(err)=%.4f', ...
        N_train_sizes(k), results(k).P_error), 'FontSize', 14);
    legend('show', 'Location', 'best', 'FontSize', 9);
    axis([-5 5 -5 5]);
    set(gca, 'FontSize', 11);
end


% Helper functions for sigmoid and negative log likelihood

% Sigmoid function
function s = sigmoid(z)
    s = 1 ./ (1 + exp(-z));
end

% Negative log-likelihood for logistic regression
function nll = negLogLikelihood(params, X_aug, y)
    p = sigmoid(X_aug * params);
    % add clipping to avoid log(0)
    p = max(p, 1e-10);
    p = min(p, 1 - 1e-10);
    nll = -sum(y .* log(p) + (1 - y) .* log(1 - p));
end