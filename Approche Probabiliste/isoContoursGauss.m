function [X,Y,Z]=isoContoursGauss(mu,Sigma)

% Aide pour l'affichage d'isocontours d'une loi en 2D

% determination du range de calcul
xmin=-2;
xmax=1.5;
ymin=-2;
ymax=1.5;

% discretisation de chaque axe sur 100 points
pasx=(xmax-xmin)/100;
x=(xmin:pasx:xmax-pasx);
pasy=(ymax-ymin)/100;
y=(ymin:pasy:ymax-pasy);

% determination des coordonnees 2D (10000 points):
% le point (i,j) de l'espace a pour coordonnee (X(i,j), Y(i,j))
[X,Y]=meshgrid(x,y);


% vectorisation des 10000 coordonnees 
XYvec=zeros(2,size(X,1)*size(X,2));
XYvec(1,:)=reshape(X,1,size(X,1)*size(X,2));
XYvec(2,:)=reshape(Y,1,size(X,1)*size(X,2));

% calcul de la distribution sur les 10000 points
Zvec=mvnpdf(XYvec',mu',Sigma);
Z=reshape(Zvec,size(X,1),size(X,2));


Nc=10; % nombre d'isocontours à afficher
% % figure,
% M=max(max(Z));
% v=M*[0.98 0.95 0.90 0.8 0.6 0.4];
contour(X,Y,Z,Nc);  