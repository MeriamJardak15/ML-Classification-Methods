close all
clear all
clc
%1. Définir les données d'entrée et de sortie en utilisant un certain nombre déchantillons pour chaque
%classe (25 par exemple).
N=25;
%2. Créer deux ensembles aléatoires de données d'entrées avec un décalage entre eux. (randn)
offset=10;
X=[randn(N,2);randn(N,2)+offset];
%3. La première classe a le label 1, tandis que la seconde a le label 0.
C=[zeros(N,1); ones(N,1)];
%4. Visualiser les échantillons d'entrée/cible. (plotpv)
figure;
plotpv(X',C')
%5. Créer et former un perceptron (perceptron, train, configure, view)
net=perceptron;
net=train(net,X',C');
view(net);
figure;
plotpv(X',C')
%6. Tracer la limite de decision (plotpc)
plotpc(net.iw{1,1},net.b{1})
%7. Tester avec xtest = [0.7; 1.2]; et visualiser la solution
xtest=[0.7;1.2];
ytest=net(xtest);