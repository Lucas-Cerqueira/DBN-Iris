%function test_example_DBN
load mnist_uint8;
load fisheriris;

str_targets = species;
% 2 - Normalization (target)
fprintf('Normalizing Target\n');
targets =zeros(size(str_targets,1),3);
for i = 1:size(str_targets,1)
    if strcmp(str_targets(i),'setosa')
        targets(i,1) = 1;
    end
    if strcmp(str_targets(i),'versicolor')
        targets(i,2) = 1;
    end
    if strcmp(str_targets(i),'virginica')
        targets(i,3) = 1;
    end
end

n_folds = 2;

CVO = cvpartition(species,'Kfold',n_folds);
for ifolds = 1%:n_folds
    trn_id =  CVO.training(ifolds);
    tst_id =  CVO.test(ifolds);
       
    itrn = []; itst = [];
    for i = 1:length(meas)
       if trn_id(i) == 1
          itrn = [itrn;i];
       else
          itst = [itst;i];
       end
    end
    ival = itst;
        
    %Min = -1 and Max = 1
    [~, norm_fact] = mapminmax(meas(itrn,:)');
    data_norm = mapminmax('apply', meas' ,norm_fact);
end
    

%train_x = double(train_x) / 255;
%test_x  = double(test_x)  / 255;

train_x = (data_norm(:,itrn) + 1)./2;
test_x = (data_norm(:,itst) + 1)./2;

%train_y = double(train_y);
%test_y  = double(test_y);

train_y = targets(itrn,:)';
test_y = targets(itst,:)';

% %%  ex1 train a 100 hidden unit RBM and visualize its weights
% rand('state',0)
% %dbn.sizes = [100];
% dbn.sizes = [75];
% opts.numepochs =   1;
% %opts.batchsize = 100;
% opts.batchsize = size (meas,1)/n_folds;
% opts.momentum  =   0;
% opts.alpha     =   1;
% dbn = dbnsetup(dbn, train_x', opts);
% dbn = dbntrain(dbn, train_x', opts);
% figure; visualize(dbn.rbm{1}.W');   %  Visualize the RBM weights


%%  ex2 train a 100-100 hidden unit DBN and use its weights to initialize a NN
rand('state',0)
%train dbn
dbn.sizes = [4 4];
opts.numepochs =   100;
opts.batchsize =    5;
opts.momentum  =    0;
opts.alpha     =    1;
dbn = dbnsetup(dbn, train_x', opts);
dbn = dbntrain(dbn, train_x', opts);

%unfold dbn to nn
nn = dbnunfoldtonn(dbn, 3);
nn.activation_function = 'sigm';

%train nn
opts.numepochs =  100;
opts.batchsize = 5;
nn = nntrain(nn, train_x', train_y', opts);
[er, bad] = nntest(nn, test_x', test_y');

assert(er < 0.10, 'Too big error');
