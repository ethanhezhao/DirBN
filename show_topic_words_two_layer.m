
function show_topic_words_two_layer(phi2, phi1, co_weight, voc, top_W, out_file, topic_weight)
%*************************************************************************
% Matlab code for
% He Zhao, Lan Du, Wray Buntine, Mingyuan Zhou, 
% "Dirichlet belief networks for topic structure learning," 
% in the thirty-second annual conference on Neural Information Processing Systems (NeurIPS) 2018.
%
% Written by He Zhao, http://ethanhezhao.github.io/
% Copyright @ He Zhao
%*************************************************************************

if ~exist('out_file','var')
    out_file = 1;
end
[~, sorted_tw_idx2] = sort(phi2, 2, 'descend');
[~, sorted_tw_idx1] = sort(phi1, 2, 'descend');
K2 = size(phi2, 1);
if exist('topic_weight','var')
    [~,sorted_topic_idx] = sort(topic_weight, 'descend');
else
    sorted_topic_idx = 1:K2;
    topic_weight = ones(K2, 1);
end

for k2 = 1:K2
    fprintf(out_file,'Main topic: ID, Weight\n');
    top_words = [];
    actual_k2 = sorted_topic_idx(k2);
    for v = 1:top_W
        top_words = [top_words, ' ', voc{sorted_tw_idx2(actual_k2, v)}];
    end
    fprintf(out_file, '%d, %f, %s\n',actual_k2, topic_weight(actual_k2), top_words);
    [~,top_co_k1s] = sort(co_weight(actual_k2, :),'descend');
    top_co_k1s = top_co_k1s(1:5);
    fprintf(out_file, 'Related topics: ID, Weight\n');
    for top_k1 = top_co_k1s
        top_words = [];
        for v = 1:top_W
            top_words = [top_words, ' ', voc{sorted_tw_idx1(top_k1, v)}];
        end
        fprintf(out_file, '%d, %0.2f, %s\n',top_k1, co_weight(actual_k2,top_k1), top_words);
    end
    fprintf(out_file,'+++++++++++++++++++++++++++++++++++++++++\n');
end
    
end


