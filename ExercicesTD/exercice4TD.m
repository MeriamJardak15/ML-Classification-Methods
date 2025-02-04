close all
clear all
clc
%1. Définir les données d'entrée et de sortie en utilisant un certain nombre d'échantillons de chaque
%catégorie (30 par exemple).
N =30;
%2. Créer quatre ensembles aléatoires de données d%entrée avec différents décalages entre eux (rand)
offset=3;
A=[rand(N,2)];
B=[rand(N,1)+offset rand(N,1)];
C=[rand(N,1) rand(N,1)+offset];
D=[rand(N,1)+offset rand(N,1)+offset];
X=[A;B;C;D];
%3. Les étiquettes prennent en consideration la position des données pour chaque classe. Définir 4
%étiquettes sur un espace 2-D.
a=[0;0];
b=[1;0];
c=[0;1];
d=[1;1];
%4. Visualiser les échantillons d%entrée/cible (plotpv). Pour les cibles, veiller à avoir la bonne
%dimension de votre matrice cible (repmat peut être utile). 
L=[repmat(a,1,N) repmat(b,1,N) repmat(c,1,N) repmat(d,1,N)]';
%5. Créer un perceptron (perceptron)
net=perceptron;
net=train(net,X',L');
%7. Visualiser la structure (view)
view(net);
%8. Tester avec xtest = [0.7; 1.2]; et visualiser la solution
xtest=[0.7;-1];
ytest=net(xtest);
figure;
plotpv(X',L')
plotpc(net.iw{1,1},net.b{1})

