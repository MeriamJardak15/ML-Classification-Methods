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




%% K-moyennes

Xns=[Xp;Xtestp] ;

opts = statset('Display','final');
[idx,C] =kmeans(Xns,K, 'Distance','cityBlock','Replicates',20,'Options',opts);
figure(10);
hold on

for k=1:K
plot (Xns(idx==k,1),Xns (idx==k,2),'O')
end
plot(C(:,1),C(:,2),'kx', 'MarkerSize',15,'LineWidth',3)
hold off ;


%% EM
Xn=[Xp;Xtestp];
N=size(Xn,1);
D=size(Xn,2);
K=4;
% Center and normalize each dimension
Xn=centrer(Xn) ;
% Initialisation des moyennes et covariances des classes
mu=C;
Sigma(1,:,:)=eye(D);
Sigma(2,:,:)=eye(D);
Sigma(3,:,:)=eye(D);
Sigma(4,:,:)=eye(D);
pi=ones(1,K)/K;
apost=zeros(K,N);
figure, scatter(Xn(:,1),Xn(:,2),20,'b','fill');
gr=[1,2,3,4]
hold;
h1=gscatter(mu(:,1),mu(:,2),gr,'rgbk','xxxx',10,[],'off');
set(h1,'MarkerSize',20)
h1=scatter(mu(1,1),mu(1,2),50,'r','O','filled');    
h2=scatter(mu(2,1),mu(2,2),50,'g','O','filled');
h3=scatter(mu (3,1),mu(3,2),50,'b','O','filled');
h4=scatter(mu(4,1),mu(4,2),50,'k', 'O','filled');
axis([-2.5 2.5 -2.5 2.5]);
% code de mise en oeuvre de l'EM (limit� � 50 it�rations)
for i=1:50
    pause(0.2)
    % Etape E
    for k=1:K
        muk=mu(k,:);
        Sigmak(:,:)=Sigma(k,:,:);
        apost(k,:)=mvnpdf(Xn,muk,Sigmak)*pi(k);
    end
    apost=apost./repmat(sum(apost),K,1);
    hold on;
    color=apost(1:3,:)';
    scatter(Xn(:,1),Xn(:,2),20,color,'fill');
    axis([-2.5 2.5 -2.5 2.5]);
    pause(0.2)
    
    % Etape M
    for k=1:K
        mu(k,:)=sum(repmat(apost(k,:)',1,D).*Xn)./repmat(sum(apost(k,:)'),1,D);
        Sigma(k,:,:)=(repmat(apost(k,:)',1,D).*(Xn-repmat(mu(k,:),N,1)))'*(Xn-repmat(mu(k,:),N,1))./(repmat(sum(apost(k,:)'),D,D));
        pi(:,k)=sum(apost(k,:))/N;
    end
    % D�cision du MAP pour affichage
    [~,IDX1]=max(apost);
    hold off;
    h1=gscatter(mu(:,1),mu(:,2),gr,'rg','xx',[],'off');
    set(h1,'MarkerSize',20);
    hold on;
    h1=scatter(mu(1,1),mu(1,2),50,'r','O','filled');    
    h2=scatter(mu(2,1),mu(2,2),50,'g','O','filled');
    h3=scatter(mu(3,1),mu(3,2),50,'b','O','filled');
    h4=scatter(mu (4,1) ,mu(4,2),50,'k', 'O', 'filled');
    color=apost(1:3,:)';
    scatter(Xn(:,1),Xn(:,2),20,color,'fill');
    axis([-2.5 2.5 -2.5 2.5]);    
end 
    
    