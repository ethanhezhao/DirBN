%*************************************************************************
% Matlab code for
% He Zhao, Lan Du, Wray Buntine, Mingyuan Zhou, 
% "Dirichlet belief networks for topic structure learning," 
% in the thirty-second annual conference on Neural Information Processing Systems (NeurIPS) 2018.
%
% Written by He Zhao, http://ethanhezhao.github.io/
% Copyright @ He Zhao
%*************************************************************************

%% prepare data and init settings
dataset_name = 'TMN';
raw_data = load(sprintf('./data/%s.mat', dataset_name));
para.train_idx = raw_data.train_idx;
para.test_idx = raw_data.test_idx;
para.train_word_prop = 0.2; % use 20% words of the training documents for training the model
para.test_word_prop = 0.5; % use one half of the words in the testing documents to estimate theta and predict the other half

para.train_burnin = 0;
para.train_collection = 3000;

ks = [100,100]; % 100 topics in the first and second layer respectively

%% run PFA+DirBN
% theta_para: parameters related to theta, i.e., doc-topic distributions
% DirBN_para: parameters related to DirBN on phi, i.e., topic-word distributions
% avg_perp_para: parameters for computing average perplexity
% zs: topic assignments for the words in the training documents
[theta_para, DirBN_para, avg_perp_para, zs] = PFA_DirBN(raw_data.x, ks, 0.05, para);


%% save the model
if ~exist('./save','dir')
    mkdir('./save');
end
save(sprintf('./save/%s_%0.1f.mat', dataset_name, para.train_word_prop),'theta_para', 'DirBN_para', 'avg_perp_para', 'zs');


%% print topic hierarchy
out_file = fopen(sprintf('./save/%s_topic_hierarchy.txt', dataset_name),'w');
show_topic_words_two_layer(DirBN_para{2}.phi,DirBN_para{1}.phi, DirBN_para{2}.beta, raw_data.voc, 10, ...
out_file,full(sum(DirBN_para{2}.n_topic_word,2)));
fclose(out_file);

