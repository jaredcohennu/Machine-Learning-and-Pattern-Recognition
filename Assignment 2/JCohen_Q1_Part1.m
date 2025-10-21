clear all, close all,

% Set all of the parameters specified in the problem
N_validate = 10000; p0 = 0.6; p1 = 0.4;
w01 = 0.5; w02 = 0.5;
mu01 = [-0.9; -1.1]; mu02 = [0.8; 0.75];
w11 = 0.5; w12 = 0.5;
mu11 = [-1.1; 0.9]; mu12 = [0.9; -0.75];
C = [0.75, 0; 0, 1.25];
Sigma01 = C;
Sigma02 = C;
Sigma11 = C;
Sigma12 = C;

% Generate labels according to priors
u_validate = rand(1, N_validate) >= p0;
N0_validate = sum(u_validate == 0);
N1_validate = sum(u_validate == 1);

% class 0 samples
n01_validate = round(N0_validate * w01);
n02_validate = N0_validate - n01_validate;
r01_validate = mvnrnd(mu01, Sigma01, n01_validate);
r02_validate = mvnrnd(mu02, Sigma02, n02_validate);
r0_validate = [r01_validate; r02_validate];

% class 1 samples
n11_validate = round(N1_validate * w11);
n12_validate = N1_validate - n11_validate;
r11_validate = mvnrnd(mu11, Sigma11, n11_validate);
r12_validate = mvnrnd(mu12, Sigma12, n12_validate);
r1_validate = [r11_validate; r12_validate];

% Combine into one dataset
x_validate = zeros(2, N_validate);
label_validate = zeros(1, N_validate);

x_validate(:, u_validate == 0) = r0_validate';
label_validate(u_validate == 0) = 0;
x_validate(:, u_validate == 1) = r1_validate';
label_validate(u_validate == 1) = 1;

Nc_validate = [N0_validate, N1_validate];


% Compute class-conditional PDFs on validation set using TRUE parameters
px_L0 = w01 * evalGaussian(x_validate, mu01, Sigma01) + ...
        w02 * evalGaussian(x_validate, mu02, Sigma02);

px_L1 = w11 * evalGaussian(x_validate, mu11, Sigma11) + ...
        w12 * evalGaussian(x_validate, mu12, Sigma12);

% Compute discriminant score (log-likelihood ratio)
disc_theo = log(px_L1) - log(px_L0);

% Theoretical optimal threshold
gamma_theo = p0 / p1;

% Decide which class
decision_theo = (disc_theo >= log(gamma_theo));

% Compute confusion matrix elements and P(error)
p00_theo = sum(decision_theo == 0 & label_validate == 0) / Nc_validate(1);
p10_theo = sum(decision_theo == 1 & label_validate == 0) / Nc_validate(1);
p01_theo = sum(decision_theo == 0 & label_validate == 1) / Nc_validate(2);
p11_theo = sum(decision_theo == 1 & label_validate == 1) / Nc_validate(2);

P_error_theo = p10_theo * p0 + p01_theo * p1;

%ROC CURVE GRAPH
figure(1), clf, hold on,

% Generate ROC curve by sweeping over range of thresholds
gammaRange_roc = logspace(-2, 2, 500);
P_TP_roc = zeros(1, length(gammaRange_roc));
P_FP_roc = zeros(1, length(gammaRange_roc));

for i = 1:length(gammaRange_roc)
    decision = (disc_theo >= log(gammaRange_roc(i)));
    P_FP_roc(i) = sum(decision == 1 & label_validate == 0) / Nc_validate(1);
    P_TP_roc(i) = sum(decision == 1 & label_validate == 1) / Nc_validate(2);
end

% Plot ROC curve
plot(P_FP_roc, P_TP_roc, 'b-', 'LineWidth', 2.5, ...
    'DisplayName', 'ROC Curve (True PDFs)');

% Mark the minimum P(error) point
plot(p10_theo, p11_theo, 'r*', 'MarkerSize', 20, 'LineWidth', 2.5, ...
    'DisplayName', sprintf('Min P(error): γ*=%.2f, P(err)=%.4f', gamma_theo, P_error_theo));

grid on,
xlabel('False Positive Rate: P(D=1|L=0)', 'FontSize', 12);
ylabel('True Positive Rate: P(D=1|L=1)', 'FontSize', 12);
title('ROC Curve: Theoretical Optimal Classifier (True PDFs)', 'FontSize', 14);
legend('show', 'Location', 'southeast', 'FontSize', 10);
axis([0 1 0 1]);
set(gca, 'FontSize', 11);

text(p10_theo + 0.12, p11_theo - 0.05, ...
    sprintf('Theoretical\nTP=%.4f\nFP=%.4f\nP(err)=%.4f', p11_theo, ...
    p10_theo, P_error_theo), 'Color', 'r', 'FontSize', 9, 'FontWeight', ...
    'bold', 'BackgroundColor', 'white', 'EdgeColor', 'red', 'LineWidth', 1);

fprintf('Figure 1: ROC curve generated with min P(error) marked.\n');

%DECISION BOUNDARY GRAPH
figure(2), clf, hold on,

% Create mesh for decision boundary
x1_range = [-4, 4];
x2_range = [-4, 4];
[X1, X2] = meshgrid(linspace(x1_range(1), x1_range(2), 300), ...
                     linspace(x2_range(1), x2_range(2), 300));
X_mesh = [X1(:)'; X2(:)'];

% Compute discriminant score on mesh using TRUE parameters
px_L0_mesh = w01 * evalGaussian(X_mesh, mu01, Sigma01) + ...
             w02 * evalGaussian(X_mesh, mu02, Sigma02);

px_L1_mesh = w11 * evalGaussian(X_mesh, mu11, Sigma11) + ...
             w12 * evalGaussian(X_mesh, mu12, Sigma12);

disc_mesh = log(px_L1_mesh) - log(px_L0_mesh);
disc_mesh = reshape(disc_mesh, size(X1));

% Plot decision boundary at theoretical optimal threshold
contour(X1, X2, disc_mesh, [log(gamma_theo), log(gamma_theo)], ...
    'k-', 'LineWidth', 2.5, 'DisplayName', 'Decision Boundary (γ*)');

% Determine correct/incorrect classifications
is_correct = (label_validate == decision_theo);

% Subsample validation points for visualization clarity
subsample_rate = 0.15; % Plot 15% of points
rng(200); % Set seed for reproducible subsampling
plot_indices = rand(1, N_validate) < subsample_rate;

idx_c0_ok = (label_validate == 0) & is_correct & plot_indices;
idx_c1_ok = (label_validate == 1) & is_correct & plot_indices;
idx_c0_err = (label_validate == 0) & ~is_correct & plot_indices;
idx_c1_err = (label_validate == 1) & ~is_correct & plot_indices;

plot(x_validate(1, idx_c0_ok), x_validate(2, idx_c0_ok), 'o', 'MarkerSize', 4, ...
    'MarkerEdgeColor', [0, 0.6, 0], 'LineWidth', 0.8, ...
    'DisplayName', 'Class 0 (Correct)');
plot(x_validate(1, idx_c1_ok), x_validate(2, idx_c1_ok), 's', 'MarkerSize', 4, ...
    'MarkerEdgeColor', [0, 0.6, 0], 'LineWidth', 0.8, ...
    'DisplayName', 'Class 1 (Correct)');
plot(x_validate(1, idx_c0_err), x_validate(2, idx_c0_err), 'o', 'MarkerSize', 6, ...
    'MarkerEdgeColor', [0.9, 0, 0], 'MarkerFaceColor', [0.9, 0, 0], 'LineWidth', 1, ...
    'DisplayName', 'Class 0 (Error)');
plot(x_validate(1, idx_c1_err), x_validate(2, idx_c1_err), 's', 'MarkerSize', 6, ...
    'MarkerEdgeColor', [0.9, 0, 0], 'MarkerFaceColor', [0.9, 0, 0], 'LineWidth', 1, ...
    'DisplayName', 'Class 1 (Error)');

xlabel('x_1', 'FontSize', 13);
ylabel('x_2', 'FontSize', 13);
title_str = sprintf('Decision Boundary: Theoretical Optimal Classifier (γ*=%.2f, P(error)=%.4f)', ...
    gamma_theo, P_error_theo);
title(title_str, 'FontSize', 12);
grid on, axis equal;
xlim(x1_range);
ylim(x2_range);
legend('show', 'Location', 'best', 'FontSize', 8);
set(gca, 'FontSize', 11);

hold off;