%Code provided from the homework
clear all, close all,
N = 10000; p0 = 0.65; p1 = 0.35;
u = rand(1,N)>=p0; N0 = length(find(u==0)); N1 = length(find(u==1));
mu0 = [-1/2;-1/2;-1/2]; Sigma0 = [1,-0.5,0.3;-0.5,1,-0.5;0.3,-0.5,1];
r0 = mvnrnd(mu0, Sigma0, N0);
figure(1), plot3(r0(:,1),r0(:,2),r0(:,3),'.b'); axis equal, hold on,
mu1 = [1;1;1]; Sigma1 = [1,0.3,-0.2;0.3,1,0.3;-0.2,0.3,1];
r1 = mvnrnd(mu1, Sigma1, N1);
figure(1), plot3(r1(:,1),r1(:,2),r1(:,3),'.r'); axis equal, hold on,

%Naive Bayesian classifier identity matrix 
Sigma_NB = eye(3);   % identity covariance for both classes

n = 3; % number of dimensions
x = zeros(n, N); % All of the data stored as columns
label = zeros(1, N); % Will store labels

Nc = [N0, N1]; % number of samples from each class
x = zeros(n,N); % save up space

%Assign the class labels
x(:, u==0) = r0';
label(u==0) = 0;
x(:, u==1) = r1';
label(u==1) = 1;

% Code modified from provided ExpectedRiskMinimization.m
lambda = [0 1; 1 0];  % loss values
gamma = (lambda(2,1) - lambda(1,1))/(lambda(1,2) - lambda(2,2)) * p0/p1; 
%threshold

discriminantScore = log(evalGaussian(x, mu1, Sigma_NB)) - log(evalGaussian ...
    (x, mu0, Sigma_NB)); % - log(gamma);


% Create an array of gamma values to sweep for ROC curve
gammaRange = logspace(-3, 10, 1000); % Sweep threshold values (-10,10 is 
% same as 0,inf in log space)
numGamma = length(gammaRange);

% Preallocate arrays for ROC curve
P_TP = zeros(1, numGamma);  % True positive rate P(D=1|L=1)
P_FP = zeros(1, numGamma); % False positive rate P(D=1|L=0)

% Create roc curve by sweeping
for i = 1:numGamma
    % Make decision based on current gamma threshold
    decision = (discriminantScore >= log(gammaRange(i)));
    
    % Find indices for each outcome (following class example pattern)
    ind00 = find(decision==0 & label==0); p00 = length(ind00)/Nc(1); % true negative
    ind10 = find(decision==1 & label==0); p10 = length(ind10)/Nc(1); % false positive
    ind01 = find(decision==0 & label==1); p01 = length(ind01)/Nc(2); % false negative
    ind11 = find(decision==1 & label==1); p11 = length(ind11)/Nc(2); % true positive
    
    % Store for ROC curve
    P_TP(i) = p11;   % P(D=1|L=1)
    P_FP(i) = p10;  % P(D=1|L=0)
end

P_error = zeros(1, numGamma);
for i = 1:numGamma %Calculate the error for each gamma value
    P_error(i) = P_FP(i) * p0 + (1 - P_TP(i)) * p1;
end

[min_P_error_empirical, min_idx] = min(P_error); %Get the minimum error
gamma_empirical = gammaRange(min_idx);
P_TP_empirical = P_TP(min_idx);
P_FP_empirical = P_FP(min_idx);

% Theoretical optimal probabilities using gamma from part 1 of part A
decision_theoretical = (discriminantScore >= log(gamma));
ind00_theo = find(decision_theoretical==0 & label==0);
ind10_theo = find(decision_theoretical==1 & label==0);
ind01_theo = find(decision_theoretical==0 & label==1);
ind11_theo = find(decision_theoretical==1 & label==1);

p00_theo = length(ind00_theo)/Nc(1);
p10_theo = length(ind10_theo)/Nc(1);
p01_theo = length(ind01_theo)/Nc(2);
p11_theo = length(ind11_theo)/Nc(2);

Perror_theo = [p10_theo, p01_theo] * Nc' / N;

% Plot ROC curve
figure(2), clf,
plot(P_FP, P_TP, 'b-', 'LineWidth', 2, "DisplayName", 'ROC Curve'); hold on,

% plot the optimal points
plot(p10_theo, p11_theo, 'ro', 'MarkerSize', 12, 'LineWidth', 2.5, ...
    'MarkerFaceColor', 'r', 'DisplayName', sprintf('Theoretical Optimal (\\gamma=%.2f)', gamma));
plot(P_FP_empirical, P_TP_empirical, 'gs', 'MarkerSize', 12, 'LineWidth', ...
    2.5, 'MarkerFaceColor', 'g', 'DisplayName', sprintf('Empirical Min P(error) (\\gamma=%.2f)', gamma_empirical));

% Plot random classifier baseline
%plot([0 1], [0 1], 'k--', 'LineWidth', 1, 'DisplayName', 'Random Classifier');

grid on,
xlabel('False Positive Rate', 'FontSize', 12);
ylabel('True Positive Rate', 'FontSize', 12);
title('ROC Curve for ERM Classifier', 'FontSize', 14);
legend('show', 'Location', 'SouthEast', 'FontSize', 10);
axis([0 1 0 1]);

% Add labels for the optimal points
text(p10_theo + 0.12, p11_theo - 0.05, ...
    sprintf('Theoretical\nTP=%.4f\nFP=%.4f\nP(err)=%.4f', p11_theo, ...
    p10_theo, Perror_theo), 'Color', 'r', 'FontSize', 9, 'FontWeight', ...
    'bold', 'BackgroundColor', 'white', 'EdgeColor', 'red', 'LineWidth', 1);
text(P_FP_empirical + 0.02, P_TP_empirical - 0.05, ...
    sprintf('Empirical\nTP=%.4f\nFP=%.4f\nP(err)=%.4f', ...
    P_TP_empirical, P_FP_empirical, min_P_error_empirical), 'Color', 'g', ...
    'FontSize', 9, 'FontWeight', 'bold', 'BackgroundColor', 'white', ...
    'EdgeColor', 'green', 'LineWidth', 1);