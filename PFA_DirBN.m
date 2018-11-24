function [theta_para, DirBN_para, avg_perp_para, zs] = PFA_DirBN(x, ks, eta, para)
%*************************************************************************
% Matlab code for
% He Zhao, Lan Du, Wray Buntine, Mingyuan Zhou, 
% "Dirichlet belief networks for topic structure learning," 
% in the thirty-second annual conference on Neural Information Processing Systems (NeurIPS) 2018.
%
% Written by He Zhao, http://ethanhezhao.github.io/
% Copyright @ He Zhao
%*************************************************************************

%% process data
data = process_data(x, para);
N = length(para.train_idx);
V = size(x,1);

%% init latent variables
zs = randi(ks(1), length(data.train_ds), 1); 
zs_ds = full(sparse(zs, data.train_ds, 1, ks(1), N));
zs_ws = full(sparse(zs, data.train_ws, 1, ks(1), V));
n_dot_k = sum(zs_ds, 2);            
DirBN_para = init_DirBN(ks, V, eta);
theta_para = init_theta(ks(1), N);
avg_perp_para = init_avg_perp();

%% sample
for iter = 1 : (para.train_burnin + para.train_collection)
    % sample topic assignments by the collapsed Gibbs sampling
    [zs_ds, zs_ws, n_dot_k, zs] = collapsed_gibbs_topic_assignment_mex(zs_ds, ...
    zs_ws,n_dot_k, zs, data.train_ws, data.train_ds, ...
    repmat(theta_para.r_k, 1, N), DirBN_para{1}.psi, sum(DirBN_para{1}.psi, 2));

    % sample DirBN
    DirBN_para = sample_DirBN(zs_ws, DirBN_para);
    
    % sample theta
    theta_para = sample_theta(zs_ds, theta_para);
    
    % computer perplexity
    if mod(iter, 5) == 0
       train_perp = compute_train_perp(data, theta_para, DirBN_para);
    end
    if iter > para.train_burnin && mod(iter, 5) == 0
        avg_perp_para.count = avg_perp_para.count + 1;    
        avg_perp_para = compute_avg_test_perp(avg_perp_para, ...
            data, theta_para, DirBN_para);
    end
    if mod(iter, 5) ==0
       fprintf('iter: %d, train perplexity: %d, avg test perplexity: %d\n', ... 
       iter, floor(train_perp), floor(avg_perp_para.test_perp(end))); 
    end
end
end

function data = process_data(x, para)

    if isempty(para.test_idx)
        x_train = x;
        x_test = [];
    else
        x_train = x(:, para.train_idx);   
        if ~isempty(para.test_idx)
            x_test = x(:, para.test_idx);
        else
            x_test = [];
        end   
    end
    [x_train_train, ~, ws, ds, train_idx, ~] = PartitionX_v1(x_train, para.train_word_prop);
    data.x_train_train = x_train_train;
    data.train_ws = ws(train_idx);
    data.train_ds = ds(train_idx);
    data.mask_train = sparse(x_train);
    data.flag_train_train = x_train_train > 0;
    if ~isempty(x_test) 
        [x_train_test, x_test_test, ~, ~, ~, ~]= PartitionX_v1(x_test, para.test_word_prop);
        data.flag_train_test = x_train_test > 0;
        data.flag_test_test = x_test_test > 0;
        data.mask_test = sparse(x_test);
        data.x_train_test = x_train_test;
        data.x_test_test = x_test_test;
    end
end

function theta_para =  init_theta(K, N)
    theta_para.p_j = (1 - exp(-1)) .* ones(1,N);
    theta_para.r_k = 1/K .* ones(K, 1);
    theta_para.gamma0 = 1; theta_para.c0 = 1;
end

function theta_para = sample_theta(theta_count, theta_para)
    b0 = 0.01; a0 = 0.01;
    t = CRT_sum_mex_matrix_v1(sparse(theta_count'), theta_para.r_k')';  
    [theta_para.r_k, theta_para.gamma0, theta_para.c0]=Sample_rk(full(t), theta_para.r_k, ...
    theta_para.p_j, theta_para.gamma0, theta_para.c0);
    theta_para.p_j = betarnd(sum(theta_count, 1) + a0, sum(theta_para.r_k, 1) + b0);
    theta_para.theta = randg(theta_para.r_k + theta_count) .* theta_para.p_j;
end

function avg_perp_para = init_avg_perp()
    avg_perp_para.test_theta = 0;
    avg_perp_para.count = 0;
    avg_perp_para.test_perp = [NaN];
    avg_perp_para.test_phi_theta = 0;    
    avg_perp_para.test_phi_theta_sum = 0;
end

function train_perp = compute_train_perp(data, theta_para, DirBN_para)
    train_phi_theta = Mult_Sparse(data.mask_train, DirBN_para{1}.phi', theta_para.theta);
    train_phi_theta = train_phi_theta ./ max(realmin, sum(theta_para.theta, 1));
    train_perp = sum(data.x_train_train(data.flag_train_train) .* log(train_phi_theta(data.flag_train_train)));
    train_perp = full(exp(-train_perp ./ sum(data.x_train_train(:))));
end

function [avg_perp_para] = compute_avg_test_perp(avg_perp_para, data, theta_para, DirBN_para)
    theta_para.test_theta = infer_theta(data.x_train_test, theta_para, DirBN_para);
    test_phi_theta = Mult_Sparse(data.mask_test,  DirBN_para{1}.phi', theta_para.test_theta);
    s_test_phi_theta = sum(theta_para.test_theta, 1);
    avg_perp_para.test_phi_theta = avg_perp_para.test_phi_theta + test_phi_theta;
    avg_perp_para.test_phi_theta_sum = avg_perp_para.test_phi_theta_sum + s_test_phi_theta;
    test_phi_theta = avg_perp_para.test_phi_theta ./ avg_perp_para.count;
    s_test_phi_theta = avg_perp_para.test_phi_theta_sum ./ avg_perp_para.count;
    test_phi_theta = test_phi_theta ./ s_test_phi_theta;
    test_perp = sum(data.x_test_test(data.flag_test_test) .* log(test_phi_theta(data.flag_test_test)));
    test_perp = exp(-test_perp ./ sum(data.x_test_test(:)));
    avg_perp_para.test_perp(end+1) = full(test_perp);
end


