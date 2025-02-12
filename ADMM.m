function [max_view_acc,max_view_gmean,max_view_fscore,max_view_auc,time] = ADMM(Train_data_A, Train_data_B, Test_data_A, Test_data_B, para, sigtau, stop, nag, theta,kfold,datanum)
%% Parameter Description:
% a, b are the coefficients within the linex loss; C1 is the weight of the loss from the A perspective; C2 is the weight of the loss from the B perspective; C3 is the weight of the difference between the two perspectives;
% rbf_sig is the parameter of the kernel; sig1-4 is the weight of the two-norm loss of the admm function; tau is the learning rate when optimizing the u variable;
% maxiter1 and tol are the maximum number of iterations and error tolerance of admm;
% maxiter2, tol2 are the maximum number of iterations and error tolerance of NAG (or GD); eta is the learning rate of NAG, and r is the weight of past impulse influence
%% Construct kernel matrix


x1=Train_data_A(:,1:end-1);
x2=Train_data_B(:,1:end-1);
Y=Train_data_B(:,end);

[n,~] = size(x1);
X1 = [x1,ones(n,1)];
[n,~] = size(x2);
X2 = [x2, ones(n,1)];

test_x1=Test_data_A(:,1:end-1);
test_x2=Test_data_B(:,1:end-1);
test_target=Test_data_B(:,end);

[n,~] = size(test_x1);
test_x1 = [test_x1,ones(n,1)];
[n,~] = size(test_x2);
test_x2 = [test_x2, ones(n,1)];


[n, ~] = size(X1);
[m, ~] = size(X2);


One = ones(1,n);
rbf_sig = para.rho;
Ka = kernel(X1,X1,'rbf',rbf_sig);
Kb = kernel(X2,X2,'rbf',rbf_sig);
% Ka = 1e-50*kernel(X1,X1,'rbf',rbf_sig);%Do not operate at the same time as standardized anti-explosion treatment, not recommended
% Kb = 1e-50*kernel(X2,X2,'rbf',rbf_sig);

%% Parameter assignment
theta1 = theta;

a = para.a;
b=para.b;
C1=para.c1;
C2=para.c2;
C3=para.c3;
lambda = para.lam;

sig1 = sigtau.sig1;
sig2 = sigtau.sig2;
sig3 = sigtau.sig3;
sig4 = sigtau.sig4;
tau1 = sigtau.tau1;
tau2 = sigtau.tau2;

maxiter1 = stop.iter1;
maxiter2 = stop.iter2;
tol = stop.tol;
tol2 = stop.tol2;

eta = nag.eta;
r = nag.r;

%% initialize

Alp=rand(n,1);
Bet=rand(m,1);
ksiA=rand(n,1);
ksiB=rand(m,1);

pi1=rand(n,1);
pi2=rand(m,1);
pi3=rand(n,1);
pi4=rand(m,1);

u1=rand(n,1);
u2=rand(m,1);
u3=rand(n,1);
u4=rand(m,1);


%% Investigate file overwrite
%% ADMM
tic;
for iter = 1 : maxiter1

    fprintf('\n***************************** ADMM iter: %d K: %d Dataset：%s ******************************\n', iter',kfold,datanum);
    fprintf('\n******** theta: %4.4e a: %4.4e  b: %4.4e C1: %4.4e C2: %4.4e C3: %4.4e rbf: %4.4e sig: %4.4e ********\n',theta,a,b,C1,C2,C3,rbf_sig,sig1);

    Alp_pre = Alp; Bet_pre = Bet; ksiA_pre = ksiA; ksiB_pre = ksiB;

    % update Alp
    %%%%%% add 1e-10*eye(size(Alp)) try to fix the badly scale
    Alp=(theta1*Ka+2*C3*(Ka'*Ka)+sig1*(Ka'*Ka))\(2*C3*Ka'*Kb*Bet-Y.*(Ka*u1)+sig1*Y.*(Ka*(One'-ksiA+pi1)));
    %             Alp = Alp/norm(Alp);   %Anti-explosion treatment, standardization
    %Replace inv(A)*b with A\b, Replace b*inv(A) with b/A

    % update Bet
    Bet=(Kb+2*C3*(Kb'*Kb)+sig2*(Kb'*Kb))\(2*C3*(Kb'*Ka)*Alp-Y.*(Kb*u2)+sig2*Y.*(Kb*(One'-ksiB+pi2)));
    %             Bet=Bet/norm(Bet);    %Anti-explosion treatment, standardization
    %             Bet = ones(m,1);     Parameters fixed

    % update ksiA
    %             [ksiA,iterA,gradA]=NAG(Ka,Alp,ksiA,Y,a,C1,pi1,pi3,u1,u3,sig1,sig3,maxiter2,r,eta,tol2);
    [ksiA,iterA,gradA]=GD(Ka,Alp,ksiA,Y,C1,pi1,pi3,u1,u3,a,lambda,sig1,sig3,maxiter2,eta,tol2);
    NAG_A.iter = iterA;
    NAG_A.grad = gradA;
    %             ksiA = 0.1*ones(n,1);    %Parameters fixed

    %update ksiB
    %[ksiB,iterB,gradB]=NAG(Kb,Bet,ksiB,Y,b,C2,pi2,pi4,u2,u4,sig2,sig4,maxiter2,r,eta,tol2);
    [ksiB,iterB,gradB]=GD(Kb,Bet,ksiB,Y,C2,pi2,pi4,u2,u4,b,lambda,sig2,sig4,maxiter2,eta,tol2);
    NAG_B.iter = iterB;
    NAG_B.grad = gradB;
    %     iter2
    %     grad

    % update pi1
    pi1=pos(u1/sig1+(Y.*(Ka*Alp)-1+ksiA));     %pos() is defined at the end
    %             pi1 = 0.5*ones(n,1);     %Parameters fixed
    % update pi2
    pi2=pos(u2/sig2+(Y.*(Kb*Bet)-1+ksiB));
    % update pi3
    pi3=pos(u3/sig3+ksiA);
    % update pi4
    pi4=pos(u4/sig4+ksiB);

    % updating multipliers
    % update u1 (multiplier)
    u1=u1+tau1*sig1*(Y.*(Ka*Alp)+ksiA-1-pi1);
    % update u2 (multiplier)
    u2=u2+tau2*sig2*(Y.*(Kb*Bet)+ksiB-1-pi2);
    % update u3 (multiplier)
    u3=u3+tau1*sig3*(ksiA-pi3);
    % update u4 (multiplier)
    u4=u4+tau2*sig4*(ksiB-pi4);




    %% variable print area
    cal = [norm(Alp),norm(Bet),norm(ksiA),norm(ksiB);...
        norm(pi1),norm(pi2),norm(pi3),norm(pi4);...
        norm(u1),norm(u2),norm(u3),norm(u4)];
    variablename = ['Alp: %4.4e \n','Bet: %4.4e \n','ksiA: %4.4e \n','ksiB: %4.4e \n'];
    fprintf(variablename, cal(1,:));
    piname = ['pi1: %4.4e \n','pi2: %4.4e \n','pi3: %4.4e \n','pi4: %4.4e \n'];
    fprintf(piname,cal(2,:));
    uname = ['u1: %4.4e \n','u2: %4.4e \n','u3: %4.4e \n','u4: %4.4e \n'];
    fprintf(uname,cal(3,:));

    %%% calculate objective value
    fval(iter) = 1/2*theta1*Alp'*Ka*Alp+1/2*Bet'*Kb*Bet...
        +C1/lambda*One*(1-1./(1+lambda*(ksiA.^2).*exp(a*Y.*ksiA)))+C2/lambda*One*(1-1./(1+lambda.*(ksiB.^2).*exp(a*Y.*ksiB)))...
        +C3*One*((Ka*Alp-Kb*Bet).*(Ka*Alp-Kb*Bet));
    res_pri_vector = [(Y.*(Ka*Alp)+ksiA-1-pi1);(Y.*(Kb*Bet)+ksiB-1-pi2);(ksiA-pi3);(ksiB-pi4)];
    res_pri(iter) = norm(res_pri_vector);

    fprintf('fval: %4.4e \n',fval(iter));
    fprintf('res_pri: %4.4e \n',res_pri(iter));

    %k=k+1;
    %end

    %% stopCond
    if isnan(norm(Alp))==1 || isnan(norm(Bet))==1
        disp(' !!!ADMM is Exploding!!! ');  break;
    end
    stopCond1 = norm(Alp - Alp_pre)/norm(Alp_pre);
    stopCond2 = norm(Bet - Bet_pre)/norm(Bet_pre);
    stopCond3 = norm(ksiA- ksiA_pre)/norm(ksiA_pre);
    stopCond4 = norm(ksiB- ksiB_pre)/norm(ksiB_pre);
    stopCond = max([stopCond1 stopCond2 stopCond3 stopCond4 ]);
    %stopCond = max([stopCond1 stopCond2]);
    %     stopPath(iter) = stopCond;       %  This variable is not used
    fprintf('stopCound1: %4.4e \n', stopCond1);
    fprintf('stopCound2: %4.4e \n', stopCond2);
    fprintf('stopCound3: %4.4e \n', stopCond3);
    fprintf('stopCound4: %4.4e \n', stopCond4);
    fprintf('ADMM stopFormulaVal: %4.4e \n', stopCond);
    if (iter> 10) &&  (stopCond < tol )
        disp(' !!!stopped by ADMM termination rule!!! ');  break;
    end
end
time=toc;

modela.w = Alp;
modela.x = X1;
modela.theta = theta1;


modelb.w = Bet;
modelb.x = X2;

[acc_A,gmean_A,fscore_A,auc_A] = predict_svm2(modela, modelb,test_x1,test_x2,test_target,rbf_sig,0);%f_A
[acc_B,gmean_B,fscore_B,auc_B] = predict_svm2(modela, modelb,test_x1,test_x2,test_target,rbf_sig,1);%f_B
[acc_W,gmean_W,fscore_W,auc_W] = predict_svm2(modela, modelb,test_x1,test_x2,test_target,rbf_sig,2);%f_weight
[acc_H,gmean_H,fscore_H,auc_H] = predict_svm2(modela, modelb,test_x1,test_x2,test_target,rbf_sig,3);%f_half

max_view_acc = maxview(acc_A,acc_B,acc_W,acc_H);
max_view_gmean = maxview(gmean_A,gmean_B,gmean_W,gmean_H);
max_view_fscore = maxview(fscore_A,fscore_B,fscore_W,fscore_H);
max_view_auc = maxview(auc_A,auc_B,auc_W,auc_H);

end