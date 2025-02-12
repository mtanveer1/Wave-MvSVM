function [result]=AUC(test_targets,output) 
% Calculate the AUC value, test_targets is the original sample label, and output is the probability of being judged as a -itive class obtained by the classifier.
% Both are row or column vectors 
[A,I]=sort(output); 
M=0;N=0; 
for i=1:length(output) 
    if(test_targets(i)==1) 
        M=M+1;    % Number of positive samples
    else 
        N=N+1;  %Number of negative samples
    end 
end 
sigma=0; 
for i=M+N:-1:1 
    if(test_targets(I(i))==1) 
        sigma=sigma+i;    %Positive sample rank addition
    end 
end
result=(sigma-(M+1)*M/2)/(M*N); 