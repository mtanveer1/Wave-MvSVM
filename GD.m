function [ksiA,iter2,grad]=GD(Ka,Alp,ksiA,Y,C1,pi1,pi3,u1,u3,a,lambda,sig1,sig3,maxiter2,eta,tol2)
% When searching for ksiB, there are the following changes：Alp-Bet, A-B, a-b, 1-2, 3-4
% ksi_p=ksi; % This variable is not used
%t=1;
%while t <=maxiter2
for iter2 = 1:maxiter2
    eta=eta/iter2;
    % grad=a*C1*Y.*(exp(a*Y.*ksiA)-1)+u1+u3+sig1*(ksiA+Y.*(Ka*Alp)-1-pi1)+sig3*(ksiA-pi3);
    grad=C1*ksiA.*exp(a*Y.*ksiA).*(2+a*ksiA.*Y)./(1+lambda*ksiA.^2.*exp(a*Y.*ksiA)).^2+u1+u3+sig1*(ksiA+Y.*(Ka*Alp)-1-pi1)+sig3*(ksiA-pi3);
    ksiA=ksiA-eta*grad;
%         f(iter2)=1/m*sum(exp(a*D*ksi)-a*D*ksi-ones(m,1))+sigma/2*(norm((D*A*w+ksi-ones(m,1)*rho-pi_1),2)^2+norm((ksi-pi_2),2)^2);  % If you do not want to generate the following graphs, you do not need this line of command.
    % t=t+1;
    %stopCond
    stopCond = max(abs(grad));
    %stopCond = norm(grad);
    if (iter2> 50) &&  (stopCond < tol2)
        % disp(' !!!stopped by termination rule!!! ');
        break;
    end
end
% Check if gradient descent converges
% figure(1),clf
% step = 1;
% plot(1:step:100,f(1:step:100));
end