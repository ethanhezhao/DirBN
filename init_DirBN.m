function DirBN_para = init_DirBN(ks, V, eta)
%*************************************************************************
% Matlab code for
% He Zhao, Lan Du, Wray Buntine, Mingyuan Zhou, 
% "Dirichlet belief networks for topic structure learning," 
% in the thirty-second annual conference on Neural Information Processing Systems (NeurIPS) 2018.
%
% Written by He Zhao, http://ethanhezhao.github.io/
% Copyright @ He Zhao
%*************************************************************************

T = length(ks);
DirBN_para = cell(T,1);
for t = 1:T
    DirBN_para{t}.psi = eta * ones(ks(t),V);
    if t > 1
        if t < T
            DirBN_para{t}.phi = randg(DirBN_para{t}.psi);
        else
            DirBN_para{t}.phi = randg(DirBN_para{t}.psi .* ones(ks(t),V));
        end
        DirBN_para{t}.phi = DirBN_para{t}.phi ./ sum(DirBN_para{t}.phi,2);
        DirBN_para{t}.beta = 0.5 .* ones(ks(t),ks(t-1));
        DirBN_para{t}.beta_gammak = 0.1 * ones(ks(t),1);
        DirBN_para{t}.beta_c = 0.1;
        DirBN_para{t}.beta_gamma0 = 1.0;
        DirBN_para{t}.beta_c0 = 1.0;
    end
end

end