function DirBN_para = sample_DirBN(n_topic_word1, DirBN_para, T_current)
%*************************************************************************
% Matlab code for
% He Zhao, Lan Du, Wray Buntine, Mingyuan Zhou, 
% "Dirichlet belief networks for topic structure learning," 
% in the thirty-second annual conference on Neural Information Processing Systems (NeurIPS) 2018.
%
% Written by He Zhao, http://ethanhezhao.github.io/
% Copyright @ He Zhao
%*************************************************************************

T = length(DirBN_para);
if nargin < 3
   T_current = T; 
end
DirBN_para{1}.n_topic_word = n_topic_word1;
if T > 1
    %% propagate the latent counts from the bottom up
    for t = 1:T_current-1
        DirBN_para = sample_DirBN_counts(DirBN_para, t);
    end
    
    %% update the latent variables from the top down
    for t = T_current:-1:1
        % update psi
        if t < T 
            DirBN_para{t}.psi =  DirBN_para{t+1}.beta' * DirBN_para{t+1}.phi;
        else
            psi = sample_DirBN_eta(DirBN_para{T}.psi(1, 1), DirBN_para{T}.n_topic_word);
            DirBN_para{T}.psi = ones(size(DirBN_para{T}.n_topic_word)) .* psi;
        end
        % update beta
        if t > 1
            DirBN_para = sample_DirBN_beta(DirBN_para, t);
        end
        % update phi
        phi = randg(DirBN_para{t}.psi + DirBN_para{t}.n_topic_word) + eps;
        phi = phi ./ sum(phi, 2);
        DirBN_para{t}.phi = phi;
    end
else
    psi = sample_DirBN_eta(DirBN_para{T}.psi(1,1), DirBN_para{T}.n_topic_word);
    DirBN_para{T}.psi = ones(size(n_topic_word1)) .* psi;
    phi = randg(DirBN_para{T}.psi + DirBN_para{T}.n_topic_word) + eps;
    phi = phi ./ sum(phi,2);
    DirBN_para{T}.phi = phi;
end
end

function eta = sample_DirBN_eta(eta, n)
%% sample a single symmetric Dirichlet

mu_0 = 0.1;
nu_0 = 10.0;
[K,V] = size(n);
log_q = -log(betarnd(V .* eta, sum(n,2)));
t = zeros(K,V);
t(n>0) = 1;
for k = 1:K
    for v = 1:V
        for j=1:n(k,v)-1
            t(k,v) = t(k,v) + double(rand() < eta ./ (eta + j));
        end
    end
end
eta = randg(mu_0 + sum(t(:))) ./ (nu_0 + V .* sum(log_q));

end