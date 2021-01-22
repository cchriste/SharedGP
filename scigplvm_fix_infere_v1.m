function model = scigplvm_fix_infere_v1(model,k,yk_star)
% fix exsiting model parameters to make predictions
% k: the k-th space yk_star lives in 

rng(1)

a0 = 1e-3; b0 = 1e-3;
N = size(model.U,1);         %origina samples 
rank = size(model.U,2);
N_star = size(yk_star,1);
yTr = model.yTr;
kerType = model.kerType;

assert( size(yk_star,2)==size(model.yTr{k},2) ,'not consistent yi_star with model')

invGPi = train_cigp_v2(model.yTr{k}, model.U, yk_star);
u_invGP = invGPi.pred_mean;

N_total = N + N_star;
U = reshape(model.params(1:N*rank), N, rank);
U_all = [U;u_invGP];
yTr{k} = [yTr{k};yk_star];

params = U_all(:);
params_length(1) = length(params);
for i = 1:length(yTr)    %number of space
    
    params = [params;log(model.ker_params{i}.l);log(model.ker_params{i}.sigma);log(model.ker_params{i}.sigma0);log(model.bta{i})];   
    params_length(i+1) = 3+rank;
end


paraCell = vec2sPara(params,params_length);
params_k = [paraCell{1}; paraCell{k+1}];
[kN,kDim] = size(yTr{k});

%optimize just for k space

% a inefficient way to get the index
% idx_use = idx_in_matrix(1:N,N_total,rank);
% mask = zeros(size(params_k));
% mask(idx_use) = 1;
% mask(end:-1:end+1-length(paraCell{k})) = 1;
% mask = ~mask;

idx_use = idx_in_matrix(N+1:N_total,N_total,rank);
mask = true(size(params_k));
mask(idx_use) = false;

params_used0 = params_k(~mask);

obj_func = @(params_k) log_evidence_cigplvm(params_k,'ard', a0, b0, kDim, kN, rank, yTr{k}*yTr{k}');
obj_func_mask = @(params_used) obj_mask(params_used, mask, params_k,obj_func) 

fastDerivativeCheck(obj_func_mask, params_used0);

opt = [];
opt.MaxIter = 1000;
opt.MaxFunEvals = 10000;
new_params_used = minFunc(obj_func_mask, params_used0, opt);

fastDerivativeCheck(obj_func_mask, new_params_used);

params_k(~mask) = new_params_used;
paraCell{1} = params_k(1:N_total*rank);

params = cell2mat(paraCell');



%% predict
    U_all = reshape(params(1:N_total*rank), N_total, rank);
    U_star = U_all(N+1:N+N_star,:);
    U = U_all(1:N,:);

    idx = N_total * rank;
    for i = 1:length(yTr)
        [ker_params{i},idx] = load_kernel_parameter(params, rank, kerType, idx);    
        bta{i} = exp(params(idx+1));
        idx = idx+1;
        
        Sigma{i} = 1/bta{i}*eye(N) + ker_func(U,ker_params{i});
        Knn{i} = ker_cross(U_star, U, ker_params{i});
        y_star{i} = Knn{i}*(Sigma{i}\yTr{i}(1:N,:));
        
    end
    
%     model = [];
    model.ker_params = ker_params;
    model.bta = bta;
    model.U = U;
    model.params = params;
    model.kerType = kerType;
%     model.train_pred = y_pred;
%     model.yTr = yTr;  %donet write
    model.y_star = y_star;
    model.u_star = U_star;
end

function [f, df] = obj_mask(params_used, mask, params_all,obj_func) 
    %partial objective function using mask flag vector
    %obj_func: a function of all parameter
    %mask: the Non-used flag vector
    %params_all(~mask) = params_used
    params_all(~mask) = params_used;
    [f, df] = obj_func(params_all);
    df = df(~mask);
end



function [f, df] = log_evidence_share_miss(params,params_length, a0, b0, yTr, dim_latent,N_total)
% likelihood for missing value 
    
    paraCell = vec2sPara(params,params_length);
    U = reshape(paraCell{1},N_total,dim_latent);
    
    df_u = zeros(size(paraCell{1}));
    f = 0;
    for i = 1:length(yTr)
        [iN,iDim] = size(yTr{i});
%         idx_use = idUse(params_length(1),1:iN,dim_latent);
        idx_use = idx_in_matrix(1:iN,N_total,dim_latent);
        
        params_i = [paraCell{1}(idx_use); paraCell{i+1}];
        [f_i, df_i] = log_evidence_cigplvm(params_i,'ard', a0, b0, iDim, iN, dim_latent, yTr{i}*yTr{i}');
        df_i_u = df_i(1:iN*dim_latent);
        df_i_hyp{i} = df_i(iN*dim_latent+1:end);
        
        df_u(idx_use) = df_u(idx_use) + df_i_u;
        f = f + f_i;
    end
    df = df_u;
    for i = 1:length(yTr)
        df = [df;df_i_hyp{i}];
    end
end


% function [f, df] = log_evidence_share_iduse(params,params_length, a0, b0, rank, ytr1,ytr2,N1,N2,dim_latent)
%     
%     
%     paraCell = vec2sPara(params,params_length);
%     params_1 = [paraCell{1}; paraCell{2}];
%     
%     idx_use = uParaIdx(paraCell{1},1:N2,N1,dim_latent);
%     params_2 = [paraCell{1}(idx_use); paraCell{3}];
%     
%     
%     [f_1, df_1] = log_evidence_hogplvm(params_1,rank,a0,b0, ytr1, 'ard', 'ard');
%     df_1_u = df_1(1:N1*dim_latent);
%     df_1_model = df_1(N1*dim_latent+1:end);
% 
%     
%     [N2,m2] = size(ytr2);
%     [f_2, df_2] = log_evidence_cigplvm(params_2, 'ard', a0, b0, m2, N2, rank(1), ytr2*ytr2');
%     
%     df_2_u = zeros(N1*dim_latent,1);
%     df_2_u(idx_use) = df_2(1:N2*dim_latent);
%     df_2_model = df_2(N2*dim_latent+1:end);
%     
%     f = f_1 + f_2;
%     df_u = df_1_u + df_2_u;
%     
%     [df, ~] = paraCell2vec({df_u, df_1_model, df_2_model});
%     
% end



function paraCell = vec2sPara(para_vec,para_length)

    for k = 1:length(para_length)
        paraCell{k} = para_vec(1:para_length(k));
        para_vec(1:para_length(k)) = [];
    end
end

function idx_used = uParaIdx(uParas,id_use,N,dim)
% u parameter vector used id 
% uParas is (N*d) matrix
% id_use is used Id, e.g., 1:N-1;
% idx_used gives the index 
    id = 1:length(uParas);
    idxm = reshape(id,N,dim);
    idxm = idxm(id_use,:);
    idx_used = idxm(:);
end

function idx = idx_in_matrix(col_id,N,dim)
% return index for a N*dim matrix using collum id id_use
    id = 1:N*dim;
    idxm = reshape(id,N,dim);
    idxm = idxm(col_id,:);
    idx = idxm(:);
end