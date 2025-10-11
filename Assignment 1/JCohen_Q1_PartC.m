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

n = 3;
% Transpose to columns
x1 = r0';
x2 = r1';
%Estimate mean vectors and covariance matrices from samples
mu0hat = mean(x1,2); S0hat = cov(x1'); mu1hat = mean(x2,2); S1hat = cov(x2');
labels = [zeros(1,N0),ones(1,N1)];

% Fisher LDA projection to get discriminant scores
Sb = (mu1hat-mu0hat)*(mu1hat-mu0hat)'; Sw = S0hat + S1hat;
[V,D] = eig(inv(Sw)*Sb); [~,ind] = sort(diag(D),'descend');
w = V(:,ind(1)); % Fisher LDA projection vector
y1 = w'*x1; y2 = w'*x2; if mean(y2)<=mean(y1), w = -w; y1 = -y1; y2 = -y2; end

% ROC curve for Fisher LDA
discriminantScoresLDA = [y1,y2];
[PfpLDA,PtpLDA,PerrorLDA,thresholdListLDA] = ROCcurve(discriminantScoresLDA,labels);
figure(2), clf,
plot(PfpLDA,PtpLDA,'b-','LineWidth',2); hold on,
xlabel('False Positive Rate','FontSize',12);
ylabel('True Positive Rate','FontSize',12);
title('ROC Curve for Fisher LDA Classifier','FontSize',14);
grid on, axis([0 1 0 1]);

% Find and mark the lowest error
[min_Perror, min_idx] = min(PerrorLDA);
Pfp_opt = PfpLDA(min_idx);
Ptp_opt = PtpLDA(min_idx);
tau_opt = thresholdListLDA(min_idx);

plot(Pfp_opt, Ptp_opt, 'gs', 'MarkerSize', 12, 'LineWidth', 2.5, ...
    'MarkerFaceColor', 'g', 'DisplayName', sprintf('Optimal (\\tau=%.2f)', tau_opt));
legend('show', 'Location', 'SouthEast', 'FontSize', 10);

text(Pfp_opt + 0.02, Ptp_opt - 0.05, ...
    sprintf('Optimal\nTP=%.4f\nFP=%.4f\nP(err)=%.4f', ...
    Ptp_opt, Pfp_opt, min_Perror), 'Color', 'g', ...
    'FontSize', 9, 'FontWeight', 'bold', 'BackgroundColor', 'white', ...
    'EdgeColor', 'green', 'LineWidth', 1);

function [Pfp,Ptp,Perror,thresholdList] = ROCcurve(discriminantScores,labels)
% Generate ROC curve by sweeping through threshold values
[sortedScores,ind] = sort(discriminantScores,'ascend');
thresholdList = [min(sortedScores)-eps,(sortedScores(1:end-1)+sortedScores(2:end))/2, max(sortedScores)+eps];
for i = 1:length(thresholdList)
    tau = thresholdList(i);
    decisions = (discriminantScores >= tau);
    Ptn(i) = length(find(decisions==0 & labels==0))/length(find(labels==0)); % True negative rate
    Pfp(i) = length(find(decisions==1 & labels==0))/length(find(labels==0)); % False positive rate
    Ptp(i) = length(find(decisions==1 & labels==1))/length(find(labels==1)); % True positive rate
    Perror(i) = sum(decisions~=labels)/length(labels); % Probability of error
end
end