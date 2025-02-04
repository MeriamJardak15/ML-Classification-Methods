close all
clear all
clc
%1. Définir 4 ensembles de données d'entrée en utilisant un certain nombre d'échantillons de chaque
%classe (100 par exemple) (rand). Visualiser les données (plot, text) 
N =100;
offset=3;
A=[rand(N,2)];
B=[rand(N,1)+offset rand(N,1)];
C=[rand(N,1) rand(N,1)+offset];
D=[rand(N,1)+offset rand(N,1)+offset];
X=[A;B;C;D];
%2. Définir le codage de sortie pour les 4 ensembles. (codage +1/-1)
a=[1;-1;-1;-1];
b=[-1;1;-1;-1];
c=[-1;-1;1;-1];
d=[-1;-1;-1;1];
%3. Préparer les entrées et les sorties pour l'apprentissage du réseau (repmat pour les cibles).
L=[repmat(a,1,N) repmat(b,1,N) repmat(c,1,N) repmat(d,1,N)]';
%4. Créer un perceptron multicouche (feedforwardnet).
net=feedforwardnet(10,'traingd');
%5. Entraîner le réseau de neurones (train).
net.divideParam.trainRatio=0.7;
net.divideParam.valRatio=0.2;
net.divideParam.testRatio=0.1;
net=train(net,X',L');
%6. Visualiser la structure (view)
view(net)
%7. Evaluer la performance du réseau et tracer les résultats en comparant la classe prédite à la classe
%cible
xtest=[0.7; 1.2];
ytest=net(xtest);
figure(1)
hold on
plot(A(:,1),A(:,2),'k+')
plot(B(:,1),B(:,2),'g*')
plot(C(:,1),C(:,2),'cd')
plot(D(:,1),D(:,2),'bx')
%évaluation sur une grille
[abs,ord]=meshgrid(-1:0.01:5);
Xd=[abs(:) ord(:)];
sortie=net(Xd');
%8. Tracer le résultat de la classification pour l'espace d'entrée complet en créant une grille (meshgrid,
%mesh, reshape)
figure(1)
m=mesh(abs,ord,reshape(sortie(1,:),length(abs),length(abs))-5);
set(m,'facecolor',[1 0 0.5],'linestyle','none')
hold on
m=mesh(abs,ord,reshape(sortie(2,:),length(abs),length(abs))-5);
set(m,'facecolor',[1 0.5 0.4],'linestyle','none')
m=mesh(abs,ord,reshape(sortie(3,:),length(abs),length(abs))-5);
set(m,'facecolor',[0.5 0.3 0.9],'linestyle','none')
m=mesh(abs,ord,reshape(sortie(4,:),length(abs),length(abs))-5);
set(m,'facecolor',[0.8 0.2 0.6],'linestyle','none')
view()
