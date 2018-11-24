function [DirBN_para] = sample_DirBN_counts(DirBN_para, t)
%*************************************************************************
% Matlab code for
% He Zhao, Lan Du, Wray Buntine, Mingyuan Zhou, 
% "Dirichlet belief networks for topic structure learning," 
% in the thirty-second annual conference on Neural Information Processing Systems (NeurIPS) 2018.
%
% Written by He Zhao, http://ethanhezhao.github.io/
% Copyright @ He Zhao
%*************************************************************************

n_topic_word = DirBN_para{t}.n_topic_word; 
phi = DirBN_para{t+1}.phi;
DirBN_beta = DirBN_para{t+1}.beta;
DirBN_psi = DirBN_para{t}.psi;
[K1,V] = size(n_topic_word);
K2 = size(phi, 1);
w_t_k2_k1 = zeros(K2, K1);
w_t_k2_v = zeros(K2, V);
for k1 = 1:K1
    for v = 1:V
        for j=1:n_topic_word(k1, v)
            if j == 1
                if_t = 1;
            else
                if_t = double(rand() < DirBN_psi(k1, v) ./ (DirBN_psi(k1, v) + j)); 
            end
           if if_t > 0
                p = phi(:, v) .* DirBN_beta(:, k1); 
                sum_cum = cumsum(p);        
                k2 = find(sum_cum > rand() * sum_cum(end),1);    
                w_t_k2_k1(k2, k1) = w_t_k2_k1(k2, k1) + 1;    
                w_t_k2_v(k2, v) = w_t_k2_v(k2, v) + 1;
            end 
        end
    end
end
DirBN_para{t+1}.n_topic_word = w_t_k2_v;
DirBN_para{t+1}.n_topic_topic = w_t_k2_k1;
end