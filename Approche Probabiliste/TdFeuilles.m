clear all
close all
clc
LeafType={'papaya','pimento','chrysanthemum','chocolate_tree'};
K=length(LeafType);
label=[];
X=[];

for LT=LeafType
    
    filenames=dir([LT{1},filesep,'Training',filesep,'*.png']);
    
    for ifile=1:length(filenames)
        
        img=imread([filenames(ifile).folder,filesep,filenames(ifile).name]);
        X=[X;extractFeatures(img)];
        label=[label,LT];
        close all;  
    end
end




%% Sélection manuelle des caractéristiques et visualisation

feat=[1 2 3]
Xs=X(:,feat);
figure(1), hold,
for LT=LeafType
    Ilt=find(strcmp(label,LT));
    scatter3(Xs(Ilt,1),Xs(Ilt,2),Xs(Ilt,3),'o','filled');
%     pause
end
legend(LeafType(1:4),'Location','SouthWest');


%% Réduction de la dimension par ACP

% codez ici l'ACP. 
% Le résultat est la matrice Xp de taille 60x36 dont les colonnes sont
% rangées dans l'ordre décroissante des composantes principales
moyX=mean(X);
X=X-moyX;
covV=X'*X;
[U,D,V]=eig(covV);
[d,ordre]= sort(diag(D),'desc');
V=V(:, ordre);
Xp=X*V;

% visualisation en 3d: on regarde 3 composantes de l'ACP

comp=[1 2 3]; % 3 premières composantes par défaut, à modifier pour visualiser d'autres composantes

figure(2), hold,
for LT=LeafType
    Ilt=find(strcmp(label,LT));
    scatter3(Xp(Ilt,comp(1)),Xp(Ilt,comp(2)),Xp(Ilt,comp(3)),'o','filled');
end
legend(LeafType(1:4),'Location','SouthWest');


% détermination de la dimension pour la suite du TD

ncp=3; % peut être modifié!
% Xp=Xp(:,1:ncp);


%% Approches supervisées
%% Non paramatrique - Parzen

% Training - codage de la fonction gaussParzen.m à réaliser!

% Visualisation du résultat en dimension 2
% isocontoursParzen fait appel à la fonction gaussParzen

sig=0.3; % écart-type du noyau gaussien

cp=[1 2]; % Choix de 2 dimensions à visualiser entre 1 et ncp

figure(3); hold on;
for LT=LeafType % boucle sur les types des classes
    Ilt=find(strcmp(label,LT)); 
    isoContoursParzen(Xp(Ilt,1:2),sig);
    scatter(Xp(Ilt,cp(1)),Xp(Ilt,cp(2)),'o','filled');
end


% Evaluation du classifieur au sens du Maximum de Vraisemblance
% sur les données d'apprentissage 


% Test sur les nouvelles données (non utilisées pour l'apprentissage)
labelTest=[];
Xtest=[];
for LT=LeafType
    
    filenames=dir([LT{1},filesep,'Test',filesep,'*.png']);
    
    
    for ifile=1:length(filenames)
        
        img=imread([filenames(ifile).folder,filesep,filenames(ifile).name]);
        Xtest=[Xtest;extractFeatures(img)];
        labelTest=[labelTest,LT];
        close all;
    end
end

Xtest=Xtest-moyX;
Xtestp=Xtest*V;


cp=[1 2]; % Choix de 2 dimensions à visualiser entre 1 et ncp

figure(3); hold on;
for LT=LeafType
    Ilt=find(strcmp(label,LT));
    Ilt1=find(strcmp(labelTest,LT));
    isoContoursParzen(Xp(Ilt,1:2),sig);
    scatter(Xtestp(Ilt1,cp(1)),Xtestp(Ilt1,cp(2)),'o','filled');
end

% Z=zeros(20,K);
% for i=1:20
%     for j=1:K
%         LT=LeafType(j);
%         Ilt=find(strcmp(label,LT));
%         Z(i,j)=gaussParzen(Xtestp(i,:)',Xp(Ilt,:),sig);% determiner la vraissemblance de la donnee xtest ( 1*3)
%     end
% end
Z=zeros(20,K);
for i=1:20
    for j=1:K
        LT=LeafType(j);
        Ilt=find(strcmp(label,LT));
        Z(i,j)=gaussParzen(Xtestp(i,:)',Xp(Ilt,:),sig);
    end
end

success=0;
fail=0;
tab=zeros(20,2);

for i=1:20
   
    [M,I]=max(Z(i,:));
    tab(i,:)=[M,I];
    if strcmp(LeafType(I),labelTest(i))==1
        success=success+1;
  
    else
        fail=fail+1;
        
    end
end
success
fail
labelTest'       







%% Paramatrique - Gaussiennes

% Training

K=length(LeafType);
muK=zeros(ncp,K); %matrice de moyenne (ncp lignes : chaque ligne contient la moyenne de classe j(k colonnes=> k classes) pour une carateristques donnée)
CovK=repmat(eye(ncp,ncp),1,1,K);
%covprim=zeros(ncp,ncp,K);

% Evaluez ici muK et covK sur les données d'apprentissage Xp
for i=1:K
    LT=LeafType(i);
    Ilt=find(strcmp(label,LT));
    muK(:,i)=mean(Xp(Ilt,1:ncp))';
    CovK(:,:,i)=cov(Xp(Ilt,1:ncp));
end


% code pour visualisation des iso-contours en 2d:

dim_visu=[1 3]; % on choisit 2 dimensions, ici par exemple la 1ère et la 3eme

figure(5); hold on;
i=1;
for LT=LeafType 
    Ilt=find(strcmp(label,LT));
    isoContoursGauss(muK(dim_visu,i),CovK(dim_visu,dim_visu,i));
    scatter(Xp(Ilt,dim_visu(1)),Xp(Ilt,dim_visu(2)),'o','filled');
    i=i+1;
end

figure(6); hold on;
i=1;
for LT=LeafType 
    Ilt=find(strcmp(labelTest,LT));
    isoContoursGauss(muK(dim_visu,i),CovK(dim_visu,dim_visu,i));
    scatter(Xtestp(Ilt,dim_visu(1)),Xtestp(Ilt,dim_visu(2)),'o','filled');
    i=i+1;
end



% Evaluation du classifieur au sens du Maximum de Vraisemblance
% sur les données d'apprentissage

fail3 = 0;
success3 = 0;
for i=1:size(Xp,1)
 XElement = Xp(i,1:ncp);
 lltt = zeros(K,1);
 for k=1:K
 lltt2(k) = mvnpdf(XElement,muK(:,k)',CovK(:,:,k));
 end
 [M2,I2] = max(lltt2);
 if strcmp(label(i),LeafType(I2))
success3 = success3 +1;
 else
 fail3 = fail3 + 1;
 end
end

% success3
% fail3



% Test sur les nouvelles données (non utilisées pour l'apprentissage)
fail1 = 0;
success1 = 0;
for i=1:size(Xtest,1)
 XtestElement = Xtestp(i,1:ncp);
 lltt = zeros(K,1);
 for k=1:K
 lltt(k) = mvnpdf(XtestElement,muK(:,k)',CovK(:,:,k));
 end
 [M,I] = max(lltt);
 if strcmp(labelTest(i),LeafType(I))
success1 = success1 +1;
 else
 fail1 = fail1 + 1;
 end
end

success1
fail1


