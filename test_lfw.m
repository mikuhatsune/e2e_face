clear, close all

LFW_DIR = '/home/zhongyy/lfw/'

setenv('GLOG_minloglevel', '3')
addpath('~/caffe/matlab')
caffe.set_mode_gpu()
caffe.set_device(1)

scale = 1.3

%%
% name2id = containers.Map;
% f = fopen('data/lfw_raw_bbox.txt','r');
% DATA = textscan(f, '%s %d %f %f %f %f');
% fclose(f);
% for k = 1:length(DATA{1})
    % name2id(DATA{1}{k}) = k;
% end
% f = fopen('../lfw-pairs.txt');
% n = fscanf(f, '%d');
% k = 0;
% % 6000 pairs in total
% pos_pairs = zeros(3000, 2);
% neg_pairs = zeros(3000, 2);
% kp = 0; kn = 0;
% for i = 1:n(1)
    % for j = 1:n(2)*2
        % l = strsplit(fgetl(f));
        % name = sprintf('%s/%s_%04s.jpg', l{1}, l{1}, l{2});
        % if j <= n(2)
            % kp = kp+1;
            % pos_pairs(kp, 1) = name2id(name);
            % name = sprintf('%s/%s_%04s.jpg', l{1}, l{1}, l{3});
            % pos_pairs(kp, 2) = name2id(name);
        % else
            % kn = kn+1;
            % neg_pairs(kn, 1) = name2id(name);
            % name = sprintf('%s/%s_%04s.jpg', l{3}, l{3}, l{4});
            % neg_pairs(kn, 2) = name2id(name);
        % end
    % end
% end

%save lfw_supp DATA name2id pos_pairs neg_pairs
%%
load lfw_supp

bestM = 0; bestX = 0; bestT = 0;

H = 128; W = 128;
%H1 = 120; W1 = 120;
H1 = 128; W1 = 128;

for M = [2:2:30]
    fprintf('%d\n', M);

    caffe.reset_all();
    model = 'train_test_rbf.prototxt';
    weights = sprintf('model/_iter_%d000.caffemodel', M)
    while ~exist(weights, 'file'), pause(60), end
    pause(1)
    net = caffe.Net(model, weights, 'test');
    net.blobs('I').reshape([W1 H1 1 2]);

    fo = fopen('log/matlab_test.log', 'a');
    fprintf(fo, '\nTIME: %s TESTING: %s\n', datestr(now), weights);

    F = net.blobs('fc5').shape;
    F = F(1);

    %% extract all features first
    features = zeros(length(DATA{1}), F);

    for i = 1:length(DATA{1})
        I = imread([LFW_DIR, DATA{1}{i}]);
        
        b = [DATA{3}(i), DATA{4}(i), DATA{5}(i), DATA{6}(i)];
        S = max(b(3:4)) * scale;
        b(1:2) = b(1:2) + (b(3:4) - S) / 2;
        b(3:4) = S;
        I = safecrop(mean(I,3), b)';
        I = imresize(I, [W,H]);
        I = I((W1-W)/2+1:(W1+W)/2, (H1-H)/2+1:(H1+H)/2);
        
        im_data = single(cat(4, I, flipud(I))) / 256;
        v = net.forward({im_data});
        features(i,:) = mean(v{1}, 2);
        
        if mod(i,1000) == 0
            fprintf('%d / %d %s\n', i, length(DATA{1}), DATA{1}{i});
        end
    end

    %% compute similarity scores
    P_l2 = -sqrt(sum( (features(pos_pairs(:,1), :) - features(pos_pairs(:,2), :)) .^ 2, 2));
    N_l2 = -sqrt(sum( (features(neg_pairs(:,1), :) - features(neg_pairs(:,2), :)) .^ 2, 2));

    f = bsxfun(@rdivide, features, sqrt(sum(features.^2, 2)));
    %P_cos = f(pos_pairs(:,1), :) * f(pos_pairs(:,2), :)';
    %N_cos = f(neg_pairs(:,1), :) * f(neg_pairs(:,2), :)';
    P_cos = (2-sqrt(sum( (f(pos_pairs(:,1), :) - f(pos_pairs(:,2), :)) .^ 2, 2)))/2;
    N_cos = (2-sqrt(sum( (f(neg_pairs(:,1), :) - f(neg_pairs(:,2), :)) .^ 2, 2)))/2;

    %% search for best threshold
    tm = 0; xm = 0;
    for x=-300:0.01:-5
        t = sum(P_l2>x) + sum(N_l2<=x);
        if t > tm
            tm = t;
            xm = x;
        end
    end
    fprintf('\nl2: %f %d %f\n', xm, tm, tm/60);
    fprintf(fo, '\nl2: %f %d %f\n', xm, tm, tm/60);
    if tm > bestT
        bestM = M;
        bestX = xm;
        bestT = tm;
    end
    T_l2 = xm;

    tm = 0; xm = 0;
    for x=0.1:0.0001:0.5
        t = sum(P_cos>x) + sum(N_cos<=x);
        if t > tm
            tm = t;
            xm = x;
        end
    end
    fprintf('cos: %f %d %f\n', xm, tm, tm/60);
    fprintf(fo, 'cos: %f %d %f\n', xm, tm, tm/60);
    if tm > bestT
        bestM = M;
        bestX = xm;
        bestT = tm;
    end
    T_cos = xm;
    %break

    fprintf(fo, 'best: %d %f %d %f\n', bestM, bestX, bestT, bestT/60);
    fclose(fo);

end
