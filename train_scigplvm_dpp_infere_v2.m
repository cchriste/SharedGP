function model = train_scigplvm_dpp_infere_v2(model,k,yi_star)
% share conditional independent gplvm
% yTr must be cell. each contain n x di matrix 
% logg:
% v2: using existing model log_evidence. 
% dpp: add Dirichlet process prior
rng(1)
a0 = 1e-3; b0 = 1e-3;
N = size(model.U,1);
rank = size(model.U,2);
N_star = size(yi_star,1);
yTr = model.yTr;
kerType = model.kerType;

assert( size(yi_star,2)==size(model.yTr{k},2) ,'not consistent yi_star with model')

% U = rand(size(yTr{1},1),rank);
% [U,~,~] = svds(yTr{1},rank);
%     %init isomap
%     options.dim_new=rank; 
%     [U,model] = Isomaps(yTr,options);

invGPi = train_cigp_v2(model.yTr{k}, model.U, yi_star);
U_star = invGPi.pred_mean;

N_all = N + N_star;

U = reshape(model.params(1:N*rank), N, rank);

U_all = [U;U_star];

yTr{k} = [yTr{k};yi_star];
% 
% 
% idx = N * rank;
% for i = 1:length(D)
%     [ker_params{i},idx] = load_kernel_parameter(params, rank, kerType, idx);    
%     bta{i} = exp(params(idx+1));
% end

model.stat.dp_phi{1}=[model.stat.dp_phi{1};rand(N_star,model.T)];

params = U_all(:);
for i = 1:length(yTr)    %number of space
    
%     log_bta{i} = log(1/var(yTr{i}(:)));
%     log_sigma{i} = 0;
%     log_sigma0{i} = log(1e-4);
%     log_l{i} = zeros(rank,1);
%     
%     mShape{i} = size(yTr{i});
    
    D{i} = yTr{i}*yTr{i}';

%     params = [params;log_l{i};log_sigma{i};log_sigma0{i};log_bta{i}];
    params = [params;log(model.ker_params{i}.l);log(model.ker_params{i}.sigma);log(model.ker_params{i}.sigma0);log(model.bta{i})];
end
% N = mShape{1}(1);


    %% add DPM parameters

%% main
    
%     fastDerivativeCheck(@(params) log_evidence_share_more(params,kerType, a0, b0, rank, yTr, N_all,model), params)
    fastDerivativeCheck(@(params)  log_evidence_share_miss(params,kerType, a0, b0, rank, yTr, N_all, model), params)
    
    nIte = 100;
    opt = [];
    opt.MaxIter = 10;
    opt.MaxFunEvals = 10000;
    
    for i = 1:nIte
        U_all = reshape(params(1:N_all*rank), N_all, rank);
        
        model = E_step(model, {U_all}, 3);
%         fastDerivativeCheck(@(params) log_evidence_share_more(params,kerType, a0, b0, rank, yTr, N_all,model), params)
        fastDerivativeCheck(@(params) log_evidence_share_miss(params,kerType, a0, b0, rank, yTr, N_all, model), params)
        
%         new_params = minFunc(@(params) log_evidence_share_more(params,kerType, a0, b0, rank, yTr, N_all,model), params, opt);
        new_params = minFunc(@(params) log_evidence_share_miss(params,kerType, a0, b0, rank, yTr, N_all, model), params, opt);
        params = new_params;

    end
    
%%
    U_all = reshape(params(1:N_all*rank), N_all, rank);
    U_star = U_all(N+1:N+N_star,:);
    
    
    idx = N_all * rank;
    for i = 1:length(D)
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
    model.yTr = yTr;
    model.y_star = y_star;
    
end

%%
function [f, df] = log_evidence_share_miss(params,kerType, a0, b0, rank, yTr, N_all, model)
% function [f, df] = log_evidence_share(params,kerType, a0, b0, rank, yTr,model)

%     [N_all,m] = size(yTr{1});
%     U = reshape(params(1:N*rank), N, rank);
    f = 0;
%     df = zeros(N*rank,1);
    df = zeros(size(params));
    
    idx = N_all*rank; 
    for i = 1:length(yTr)
        iparams = [params(1:N_all*rank); params(idx+1:idx+3+rank)];
        [iN,im] = size(yTr{i});
        useIdx = 1:iN;
        
%         [f_i, df_i] = log_evidence_cigplvm(iparams,kerType, a0, b0, m, N, rank, yTr{i}*yTr{i}');
        [f_i, df_i] = log_evidence_cigplvm_miss(iparams,kerType, a0, b0, im, N_all, rank, yTr{i}*yTr{i}', useIdx);
        
        
        
        f = f + f_i;
        df(1:N_all*rank) = df(1:N_all*rank) + df_i(1:N_all*rank);
        df(idx+1:idx+3+rank) = df_i(N_all*rank+1:end);
        
        idx = idx+3+rank;
    end

    %dpp
%     idx = 0;
%     U = reshape(params(idx+1:idx+N_all*rank), N_all, rank);
%     
%     k = 1;
%     logL = -0.5*model.dp_lam(k)*  norm((U - model.stat.dp_phi{k}*model.stat.eta_mean{k}),'fro')^2;
%     dU = -model.dp_lam(k)* (U - model.stat.dp_phi{k}*model.stat.eta_mean{k});
%     
%     f_i = f_i - logL;
%     df_i(1:N_all*rank) = df_i(1:N_all*rank) - dU(:);

    %dpp
    idx = 0;
    U = reshape(params(idx+1:idx+N_all*rank), N_all, rank);
    
    k = 1;
    logL = -0.5*model.dp_lam(k)*  norm((U - model.stat.dp_phi{k}*model.stat.eta_mean{k}),'fro')^2;
    dU = -model.dp_lam(k)* (U - model.stat.dp_phi{k}*model.stat.eta_mean{k});
    
    f_i = f_i - logL;
    df_i(1:N_all*rank) = df_i(1:N_all*rank) - dU(:);
end



%%
function [f, df] = log_evidence_share_more(params,kerType, a0, b0, rank, yTr, N_all, model)

    [N,m] = size(yTr{1});
%     N_all = N_all;

    for i = 1:length(yTr)
        [NList(i), mList(i)] = size(yTr{i}); 
    end
    N_all = max(NList);

    
    U = reshape(params(1:N_all*rank), N_all, rank);
    idxNU = reshape([1:N_all*rank], N_all, rank);   %index matrix for U
    
    f = 0;
    df = zeros(size(params));
    idx = N_all*rank; 
    
    
    for i = 1:length(yTr)
        [iN,im] = size(yTr{i});
        
        idxiNU = idxNU(1:iN,:);     %i index matrix for iU
        idxiNU = idxiNU(:);         
%         iU = U(1:iN,:);
%         iparams = [iU(:); params(idx+1:idx+3+rank)];
        iparams = [params(idxiNU); params(idx+1:idx+3+rank)];
        [f_i, df_i] = log_evidence_cigplvm(iparams,kerType, a0, b0, im, iN, rank, yTr{i}*yTr{i}'); 
        
        f = f + f_i;
        df(idxiNU) = df(idxiNU) + df_i(1:iN*rank);
%         df(1:N*rank) = df(1:N*rank) + df_i(1:N*rank);
        df(idx+1:idx+3+rank) = df_i(iN*rank+1:end);
        
        idx = idx+3+rank;
    end
%     disp(f)

%   dpp
    idx = 0;
    U = reshape(params(idx+1:idx+N_all*rank), N_all, rank);
    
    k = 1;
    logL = -0.5*model.dp_lam(k)*  norm((U - model.stat.dp_phi{k}*model.stat.eta_mean{k}),'fro')^2;
    dU = -model.dp_lam(k)* (U - model.stat.dp_phi{k}*model.stat.eta_mean{k});
    
    f_i = f_i - logL;
    df(1:N_all*rank) = df(1:N_all*rank) - dU(:);


end

%%


% function [f, df] = log_evidence_share_moreInK2(params,kerType, a0, b0, rank, yTr,model,k,y_k_star)
% 
%     N_star = size(y_k_star,1);
%     [N,m] = size(yTr{1});
%     f = 0;
%     df = zeros(size(params));
%     
%     idx = N*rank; 
%     for i = 1:length(yTr)
%         
%         if i==k
%             iparams = [params(1:N*rank);params(end-N_star+1:end);params(idx+1:idx+3+rank)];
%             [f_i, df_i] = log_evidence_cigplvm(iparams,kerType, a0, b0, m, N, rank, [yTr{i};y_k_star]*[yTr{i};y_k_star]');
%             f = f + f_i;
%             df(1:N*rank) = df(1:N*rank) + df_i(1:N*rank);
%             df(end-N_star+1:end) = df(end-N_star+1:end) + df_i(N*rank+1:N*rank);    %the new
%             
%             df(idx+1:idx+3+rank) = df_i((N+N_star)*rank+1:end);
%             idx = idx+3+rank;
%         else
%             iparams = [params(1:N*rank); params(idx+1:idx+3+rank)];
% 
%             [f_i, df_i] = log_evidence_cigplvm(iparams,kerType, a0, b0, m, N, rank, yTr{i}*yTr{i}');
% 
%             f = f + f_i;
%             df(1:N*rank) = df(1:N*rank) + df_i(1:N*rank);
%             df(idx+1:idx+3+rank) = df_i(N*rank+1:end);
%             idx = idx+3+rank;
%         end
%     end
% 
%     %dpp
%     idx = 0;
%     U = reshape(params(idx+1:idx+N*rank), N, rank);
%     
%     k = 1;
%     logL = -0.5*model.dp_lam(k)*  norm((U - model.stat.dp_phi{k}*model.stat.eta_mean{k}),'fro')^2;
%     dU = -model.dp_lam(k)* (U - model.stat.dp_phi{k}*model.stat.eta_mean{k});
%     
%     f_i = f_i - logL;
%     df_i(1:N*rank) = df_i(1:N*rank) - dU(:);
% end





% function [f, df] = log_evidence_share(params,kerType, a0, b0, rank, yTr,model)
% 
%     [N,m] = size(yTr{1});
% %     U = reshape(params(1:N*rank), N, rank);
%     f = 0;
% %     df = zeros(N*rank,1);
%     df = zeros(size(params));
%     
%     idx = N*rank; 
%     for i = 1:length(yTr)
%         iparams = [params(1:N*rank); params(idx+1:idx+3+rank)];
%         
%         [f_i, df_i] = log_evidence_cigplvm(iparams,kerType, a0, b0, m, N, rank, yTr{i}*yTr{i}');
%         
%         
%         
%         f = f + f_i;
%         df(1:N*rank) = df(1:N*rank) + df_i(1:N*rank);
%         df(idx+1:idx+3+rank) = df_i(N*rank+1:end);
%         
%         idx = idx+3+rank;
%     end
% 
%     %dpp
%     idx = 0;
%     U = reshape(params(idx+1:idx+N*rank), N, rank);
%     
%     k = 1;
%     logL = -0.5*model.dp_lam(k)*  norm((U - model.stat.dp_phi{k}*model.stat.eta_mean{k}),'fro')^2;
%     dU = -model.dp_lam(k)* (U - model.stat.dp_phi{k}*model.stat.eta_mean{k});
%     
%     f_i = f_i - logL;
%     df_i(1:N*rank) = df_i(1:N*rank) - dU(:);
% end


function model = E_step(model,U, nIte)

    %fix model
    nmod = 1;
%     nIte = 5;
%     U = model.U;
    nvec = size(U{1},1);

    for iter = 1:nIte
        %update q(eta), the cluster centers sampled from base measure
        for k=1:nmod    
            model.stat.eta_cov{k} = 1./(1 + model.dp_lam(k)*sum(model.stat.dp_phi{k},1));
            model.stat.eta_mean{k} = model.stat.dp_phi{k}'*U{k}./(1/model.dp_lam(k) + repmat(sum(model.stat.dp_phi{k},1)',1,size(U{k},2)) );
            model.stat.eta_var{k} = model.dim*model.stat.eta_cov{k} + sum(model.stat.eta_mean{k}.^2,2)';
        end

        %update q(Z), indicators sampled from stick-breaking process
        for k=1:nmod
            trunc_no = model.T(k);
            sum_t = zeros(1,trunc_no);
            ex_logv2 = model.stat.ex_logv{k}(:,2);
            for t=2:trunc_no
                sum_t(t) = sum_t(t-1) + ex_logv2(t-1);
            end
            for n=1:nvec(k)
                model.dp_logphi{k}(n,:) = model.stat.ex_logv{k}(:,1)' + sum_t - 0.5*model.dp_lam(k)*model.stat.eta_var{k} ...
                    + model.dp_lam(k)*U{k}(n,:)*model.stat.eta_mean{k}';
                model.stat.dp_phi{k}(n,:) = exp_sum_dist(model.dp_logphi{k}(n,:));
            end
        end

        %update q(V), for stick-breaking process construction
        for k=1:nmod
            trunc_no = model.T(k);
            model.dp_ga{k}(:,1) = 1 + sum(model.stat.dp_phi{k}, 1)';    
            for t=1:trunc_no
                model.dp_ga{k}(t,2) = model.dp_alpha + sum(sum(model.stat.dp_phi{k}(:,t+1:end), 1));
            end
            disum = psi(sum(model.dp_ga{k},2));
            model.stat.ex_logv{k} = [psi(model.dp_ga{k}(:,1)) - disum, psi(model.dp_ga{k}(:,2)) - disum];    
        end

        %update q(lam), for inverse variance of DP generating latent factors
        if model.update_inv_cov
            for k=1:nmod
                model.dp_lam_a(k) = model.dp_lam_a0(k) + 0.5*nvec(k)*model.dim;
                model.dp_lam_b(k) = model.dp_lam_b0(k) + 0.5*(trace(U{k}*U{k}') + sum(model.stat.dp_phi{k}*model.stat.eta_var{k}')) ...
                                    - sum(sum(model.stat.eta_mean{k}*U{k}'.*model.stat.dp_phi{k}'));
                model.dp_lam(k) = model.dp_lam_a(k)/model.dp_lam_b(k);                
            end
        end

    end
    
end

