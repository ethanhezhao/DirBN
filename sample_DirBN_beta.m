function DirBN_para = sample_DirBN_beta(DirBN_para, t)
%*************************************************************************
% Matlab code for
% He Zhao, Lan Du, Wray Buntine, Mingyuan Zhou, 
% "Dirichlet belief networks for topic structure learning," 
% in the thirty-second annual conference on Neural Information Processing Systems (NeurIPS) 2018.
%
% Written by He Zhao, http://ethanhezhao.github.io/
% Copyright @ He Zhao
%*************************************************************************

a0 = 0.01; b0 = 0.01; e0 = 1; f0 = 1;
beta_gamma0 = DirBN_para{t}.beta_gamma0;
beta_gammak = DirBN_para{t}.beta_gammak;
beta_c0 = DirBN_para{t}.beta_c0;
beta_c = DirBN_para{t}.beta_c;
w_log_inv_q = -log(betarnd(sum(DirBN_para{t-1}.psi, 2), sum(DirBN_para{t-1}.n_topic_word, 2)));
w_t_k2_k1 = DirBN_para{t}.n_topic_topic;
[K2, K1] = size(w_t_k2_k1);
w_tt = zeros(K2, K1);
w_tt(w_t_k2_k1 > 0) = 1;
for k2 = 1:K2
    for k1 = 1:K1
        for j = 1:w_t_k2_k1(k2, k1) - 1
            w_tt(k2,k1) = w_tt(k2,k1) + double(rand() < beta_gammak(k2) ./ (beta_gammak(k2) + j));
        end
    end
end
w_tt_k2_dot = sum(w_tt,2);
active_k1 = ~isnan(w_log_inv_q) & ~isinf(w_log_inv_q) & w_log_inv_q ~=0;
a_K1 = sum(active_k1);
temp = log(1 + w_log_inv_q ./ beta_c);
temp = sum(temp(active_k1));
beta_gammak = randg(beta_gamma0/K2 + w_tt_k2_dot) ./ (beta_c0 + temp);
w_tt_k2_dot_t = zeros(K2, 1);
w_tt_k2_dot_t(w_tt_k2_dot > 0) = 1;
for k2 = 1:K2
    for j=1:w_tt_k2_dot(k2)-1
        w_tt_k2_dot_t(k2) = w_tt_k2_dot_t(k2) + double(rand() < (beta_gamma0/K2) ./ (beta_gamma0/K2 + j));
    end
end
beta_gamma0 = randg(a0 + sum(w_tt_k2_dot_t)) ./ (b0 + log(1 + temp ./ beta_c0));
beta_c0 = randg(e0 + beta_gamma0) ./ (f0 + sum(beta_gammak));
beta_c = randg(1.0 + a_K1 .* sum(beta_gammak) ) ./ (1.0 + sum(DirBN_para{t}.beta(:)));
DirBN_beta = randg(w_t_k2_k1 + beta_gammak) ./ (beta_c + repmat(w_log_inv_q, 1, K2)'); 
DirBN_para{t}.beta_gammak = beta_gammak;
DirBN_para{t}.beta(:, active_k1) = DirBN_beta(:, active_k1);
DirBN_para{t}.beta_c = beta_c;
DirBN_para{t}.beta_gamma0 = beta_gamma0;
DirBN_para{t}.beta_c0 = beta_c0;

end


