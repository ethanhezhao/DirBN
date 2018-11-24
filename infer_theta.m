function avg_theta = infer_theta(Xtrain_test, theta_para, DirBN_para)
%*************************************************************************
% Matlab code for
% He Zhao, Lan Du, Wray Buntine, Mingyuan Zhou, 
% "Dirichlet belief networks for topic structure learning," 
% in the thirty-second annual conference on Neural Information Processing Systems (NeurIPS) 2018.
%
% Written by He Zhao, http://ethanhezhao.github.io/
% Copyright @ He Zhao
%*************************************************************************

test_burnin = 100;
test_collection = 1;
[~, N] = size(Xtrain_test);
Xtrain_test = sparse(Xtrain_test);
p_j = median(theta_para.p_j);
p_j = p_j .* ones(1,N);
theta = max(randg(theta_para.r_k) .* p_j ./ (1 - p_j) , 1e-2);
avg_theta = 0;
b0 = 0.01; a0 = 0.01;
for iter = 1: test_burnin + test_collection
    theta_count = Multrnd_Matrix_mex_fast_v1(Xtrain_test,DirBN_para{1}.phi', theta);
    p_j = betarnd(sum(theta_count, 1) + a0, sum(theta_para.r_k, 1) + b0);
    theta = randg(theta_para.r_k + theta_count) .*  p_j;
    if iter > test_burnin
        avg_theta = avg_theta + bsxfun(@rdivide,theta,max(sum(theta,1),realmin)) / test_collection;
    end
end
end