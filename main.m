clc;
clear;
warning off
Directory=dir('.\Data\*.mat');
l=length(Directory);

for u=1:l
    t=strcat('.\Data\',Directory(u).name);
    file{u}=load(t);

    data=file{1,u};
    x1=data.X1;
    x2=data.X2;
    y=data.y;

    x1t = mapminmax(x1',0,1);
    XA = x1t';
    x2t = mapminmax(x2',0,1);
    XB = x2t';

    [m1,~]=size(XA);
    for i=1:m1
        if y(i)==0
            y(i)=-1;
        end
    end
    idx=randperm(size(XA, 1));
    XA=[XA,y];
    XB=[XB,y];
    XA = XA(idx, :);
    XB = XB(idx, :);

    cv = cvpartition(m1,'HoldOut',0.3);
    idx = cv.test;
    % Separate to training and test data
    XA_train = XA(~idx,:);
    XA_test  = XA(idx,:);
    XB_train = XB(~idx,:);
    XB_test  = XB(idx,:);



    [tau,eta] = deal(1.618,0.01);
    sig = 3.8*1e-5;
    tol = 1e-3;

    para.a = 0.5;
    para.lam = 0.2;
    para.c1 = 0.1;
    para.rho = 1;
    theta = 1;

    para.c2=para.c1;
    para.c3=para.c1;
    para.b=para.a;
    [sigtau.sig1,sigtau.sig2,sigtau.sig3,sigtau.sig4] = deal(sig);
    [sigtau.tau1,sigtau.tau2] = deal(tau);
    [stop.iter1,stop.iter2,stop.tol,stop.tol2] = deal(1000,1000,tol,0.001);   %Increase the maximum number of admm iterations iter1
    [nag.eta,nag.r] = deal(eta);


    %% testing and training
    [accuracy, ~, ~,~, time] = ADMM(XA_train, XB_train, XA_test, XB_test, para, sigtau, stop, nag, theta,part,t);
    accuracy.value

end










