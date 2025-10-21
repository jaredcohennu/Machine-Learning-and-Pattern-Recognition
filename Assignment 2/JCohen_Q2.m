clear; close all; clc;

% Generate training and validation datasets
Ntrain = 100;
Nvalidate = 1000;
[xTrain, yTrain, xValidate, yValidate] = hw2q2(Ntrain, Nvalidate);

function X = createDesignMatrix(x)
    x1 = x(1,:);
    x2 = x(2,:);
    X = [ones(1, size(x,2));x1;x2;x1.^2;x1.*x2;x2.^2;x1.^3;(x1.^2).*x2;x1.*(x2.^2);x2.^3];
    X = X';
end

% Implement ML
XTrain = createDesignMatrix(xTrain);
XValidate = createDesignMatrix(xValidate);
w_ML = (XTrain' * XTrain) \ (XTrain' * yTrain');
yPred_ML = XValidate * w_ML;
MSE_ML = mean((yValidate' - yPred_ML).^2);
fprintf('ML Validation MSE: %.4f\n', MSE_ML);

%Implement MAP
residuals_ML = yTrain' - XTrain * w_ML;
sigma2_estimate = var(residuals_ML);
fprintf('Estimated sigma^2: %.4f\n', sigma2_estimate);

% Create gamma sweep
gamma_values = logspace(-10, 10, 21);
num_gamma = length(gamma_values);

w_MAP_all = zeros(10, num_gamma);
MSE_MAP_train = zeros(1, num_gamma);
MSE_MAP_validate = zeros(1, num_gamma);

% Sweep the gamma values
for i = 1:num_gamma
    gamma = gamma_values(i);
    lambda = sigma2_estimate / gamma;
    
    % MAP
    I = eye(10);
    w_MAP = (XTrain' * XTrain + lambda * I) \ (XTrain' * yTrain');
    w_MAP_all(:, i) = w_MAP;
    
    % Evaluate on the two datasets
    yPred_train = XTrain * w_MAP;
    MSE_MAP_train(i) = mean((yTrain' - yPred_train).^2);
    yPred_validate = XValidate * w_MAP;
    MSE_MAP_validate(i) = mean((yValidate' - yPred_validate).^2);
end

% Plot Validation MSE vs Gamma
figure(1);
semilogx(gamma_values, MSE_MAP_validate, 'b-o', 'LineWidth', 2, 'MarkerSize', 6);
hold on;
semilogx(gamma_values, MSE_ML * ones(size(gamma_values)), 'r--', 'LineWidth', 2);
grid on;
xlabel('\gamma (Prior Variance)', 'FontSize', 12);
ylabel('Validation MSE', 'FontSize', 12);
title('MAP Validation Error vs Hyperparameter \gamma', 'FontSize', 14);
legend('MAP', 'ML (gamma \rightarrow \infty)', 'Location', 'best');
hold off;

% Get the lambda values and optimal gamma
lambda_values = sigma2_estimate ./ gamma_values;
[min_MSE, idx_optimal] = min(MSE_MAP_validate);
gamma_optimal = gamma_values(idx_optimal);
lambda_optimal = lambda_values(idx_optimal);
w_optimal = w_MAP_all(:, idx_optimal);

% Plot predictions for ML and MAP
figure(2);
subplot(1,2,1);
scatter3(xValidate(1,:), xValidate(2,:), yValidate, 10, 'b', 'filled');
hold on;
scatter3(xValidate(1,:), xValidate(2,:), yPred_ML, 10, 'r', 'filled');
xlabel('x_1'); ylabel('x_2'); zlabel('y');
title('ML Predictions vs True Values');
legend('True', 'ML Predictions');
grid on;

subplot(1,2,2);
scatter3(xValidate(1,:), xValidate(2,:), yValidate, 10, 'b', 'filled');
hold on;
yPred_optimal = XValidate * w_optimal;
scatter3(xValidate(1,:), xValidate(2,:), yPred_optimal, 10, 'g', 'filled');
xlabel('x_1'); ylabel('x_2'); zlabel('y');
title('Optimal MAP Predictions vs True Values');
legend('True', 'Optimal MAP Predictions');
grid on;