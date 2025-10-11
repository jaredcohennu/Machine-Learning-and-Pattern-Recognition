function out = LDAwithROCcurve(in)
out=NaN;

n = 2; 
% Generate n-dimensional data vectors from 2 Gaussian pdfs
N1 = 100; mu1 = -1e-1*ones(n,1); A1 = 3*(rand(n,n)-0.5); %S1 = A*A';
N2 = 75; mu2 = 1e-1*ones(n,1); A2 = 2*(rand(n,n)-0.5);
x1 = A1*rand(n,N1)+mu1*ones(1,N1); x2 = A2*rand(n,N2)+mu2*ones(1,N2);
% Estimate mean vectors and covariance matrices from samples
mu1hat = mean(x1,2); S1hat = cov(x1'); mu2hat = mean(x2,2); S2hat = cov(x2');
labels = [zeros(1,N1),ones(1,N2)];

% Fisher LDA projection to get discriminant scores
Sb = (mu1hat-mu2hat)*(mu1hat-mu2hat)'; Sw = S1hat + S2hat;
[V,D] = eig(inv(Sw)*Sb); [~,ind] = sort(diag(D),'descend');
w = V(:,ind(1)); % Fisher LDA projection vector
y1 = w'*x1; y2 = w'*x2; if mean(y2)<=mean(y1), w = -w; y1 = -y1; y2 = -y2; end,
% Plot the data before and after linear projection
figure(1), clf,
subplot(2,2,1), plot(x1(1,:),x1(2,:),'r*'); hold on;
plot(x2(1,:),x2(2,:),'bo'); axis equal, 
xlabel('x_1'), ylabel('x_2'),
subplot(2,2,2), plot(y1(1,:),zeros(1,N1),'r*'); hold on;
plot(y2(1,:),zeros(1,N2),'bo'); axis equal,
% ROC curve for Fisher LDA
discriminantScoresLDA = [y1,y2];
[PfpLDA,PtpLDA,PerrorLDA,thresholdListLDA] = ROCcurve(discriminantScoresLDA,labels);
subplot(2,2,3), plot(PfpLDA,PtpLDA,'og'), 
xlabel('P(False+)'),ylabel('P(True+)'), title('ROC Curve for LDA Discriminant Scores'),
subplot(2,2,4), plot(thresholdListLDA,PerrorLDA,'og'),
xlabel('Thresholds'), ylabel('P(error) for LDA Discriminant Scores'),

% ERM with Gaussian class conditional pdfs
x = [x1,x2]; g1 = evalGaussian(x,mu1hat,S1hat); g2 = evalGaussian(x,mu2hat,S2hat);
discriminantScoresERM = log(g2./g1);
subplot(2,2,2), hold on, plot(discriminantScoresERM,10*ones(1,length(labels)),'.m');
[PfpERM,PtpERM,PerrorERM,thresholdListERM] = ROCcurve(discriminantScoresERM,labels);
subplot(2,2,3), hold on, plot(PfpERM,PtpERM,'*m'),
xlabel('P(False+)'),ylabel('P(True+)'), title('ROC Curve for ERM Discriminant Scores'),
subplot(2,2,4), hold on, plot(thresholdListERM,PerrorERM,'*m'),
xlabel('Thresholds'), ylabel('P(error) for ERM Discriminant Scores'),

function [Pfp,Ptp,Perror,thresholdList] = ROCcurve(discriminantScores,labels)
[sortedScores,ind] = sort(discriminantScores,'ascend');
thresholdList = [min(sortedScores)-eps,(sortedScores(1:end-1)+sortedScores(2:end))/2, max(sortedScores)+eps];
for i = 1:length(thresholdList)
    tau = thresholdList(i);
    decisions = (discriminantScores >= tau);
    Ptn(i) = length(find(decisions==0 & labels==0))/length(find(labels==0));
    Pfp(i) = length(find(decisions==1 & labels==0))/length(find(labels==0));
    Ptp(i) = length(find(decisions==1 & labels==1))/length(find(labels==1));
    Perror(i) = sum(decisions~=labels)/length(labels);
end





