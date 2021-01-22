% Demo_viz
clear 

addpath(genpath('./L-BFGS-B-C'))
addpath(genpath('./lightspeed'))
addpath(genpath('./minFunc_2012'))
addpath(genpath('./util'))
addpath(genpath('./ISOMAPS'))
%% prepare data
load('xs.mat')

IMG_SIZE1 = 60;
IMG_SIZE2 = 60;

y4 = design_parameters';
y1 = vmis_stress';
y2 = xs';
y3 = compliance';

ifUse = logical(mod(y4(:,1),2)) & y4(:,1)<20 & y4(:,1)>-20;  %reduce data size
y4 = y4(ifUse,:);
y1 = y1(ifUse,:);
y2 = y2(ifUse,:);
y3 = y3(ifUse,:);

nTr = 200;
nTe = 100;
idAll = randperm(size(y4,1));

idTr = idAll(1:nTr);
idTe = idAll(size(y4,1):-1:size(y4,1)+1-nTe);

Y{1} = y1(idTr,:);
Y{2} = y2(idTr,:);
Y{3} = y3(idTr,:);
Y{4} = y4(idTr,:);

%%%%
rank = 2; %latent dimension
model_dpp = train_scigplvm_dpp_v2(Y,rank,'ard');
% model_dpp2 = train_scigplvm_dpp_infere_v4(model_dpp,2,y2(idTe,:));

%% viz
marker = {'r*','bo','k^','gd'};
% colormap
[val,loc] = max(model_dpp.stat.dp_phi{1, 1}');
marker = parula(max(loc));

%% scatter plot
figure(1)
clf
hold on 
for i = 1:max(loc)
%     scatter(model2.U((loc==i),1),model2.U((loc==i),2),marker{i});
    scatter(model_dpp.U((loc==i),1),model_dpp.U((loc==i),2),10,marker(i,:));
%     scatter3(model2.U((loc==i),1),model2.U((loc==i),2),model2.U((loc==i),3), 10,marker(i,:));
end


%% no tsne
cate = model_dpp.stat.dp_phi{1};
[~,labels] = max(cate,[],2);

z = model_dpp.U;
images = Y{2};

MyColor = hsv(length(unique(labels)));
% color = jet(length(unique(labels)));
figure(2)
clf
% gscatter(z(:,1),z(:,2),labels,'doleg','off')
gscatter(z(:,1),z(:,2),labels,MyColor,[],[20],'off')
hold on 
% scatter(z(:,1),z(:,2),20,MyColor(labels,:),'fill','MarkerEdgeColor',[0,0,0],'LineWidth',1.5)
scatter(z(:,1),z(:,2),80,'MarkerEdgeColor',[0,0,0],'LineWidth',2)
hold off
% 

set(gcf,'Position',[101 101 600 600])
set(gca,'position',[0 0 1 1],'units','normalized')
box on 
set(gca,'linewidth',3)

grid on 
hold on
%

% add colored domain
ifAddBoundary = true;
if ifAddBoundary  
    classifier =fitcknn(z,labels);
    X=z;
    nX = 200;
    nY = 200;
    x1range = linspace(min(X(:,1))-0.05,max(X(:,1))+0.05,nX);
    x2range = linspace(min(X(:,2))-0.05,max(X(:,2))+0.05,nY);
    % x1range = min(X(:,1)):.001:max(X(:,1));
    % x2range = min(X(:,2)):.001:max(X(:,2));
    [xx1, xx2] = meshgrid(x1range,x2range);
    XGrid = [xx1(:) xx2(:)];
    predictedspecies = predict(classifier,XGrid);

    % subplot(2,2,i);
    scatter1 = gscatter(xx1(:), xx2(:), predictedspecies,MyColor*1);
    % Set property MarkerFaceAlpha and MarkerEdgeAlpha to <1.0
%     scatter1.MarkerFaceAlpha = .2;
%     scatter1.MarkerEdgeAlpha = .2;
    
    
%     alpha(1)
    % title(classifier_name{i})
    legend off, axis tight
end


% add image to the figure
N_img = 64; %number of image to show

imgH = range(z(:,1))/15;
imgW = range(z(:,2))/15;
% imgW = imgH;

hold on 
imgShowId = randperm(size(z,1), N_img);
for i = 1:length(imgShowId)
    id = imgShowId(i);
%     img = squeeze(ytr.data(id,:,:));
    img = reshape(images(id,:),IMG_SIZE1, IMG_SIZE2)*10;
%     imagesc(xs,ys,imrotate(im',90))
        
    xs=linspace(z(id, 1) - imgH/2, z(id, 1) + imgH/2, size(img, 2) );
    ys=linspace(z(id, 2) - imgW/2, z(id, 2) + imgW/2, size(img, 1) );
      
    colormap gray;
    imagesc(xs, ys, flip(img))
%     set(gca,'dataAspectRatio',[0.01 0.01 0.01])
end
hold off