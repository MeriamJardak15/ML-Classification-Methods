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
%5. Implémenter l'algorithme de mise à jour des poids pour créer un perceptron (avec un biais).
% algorithme du gradient descendant
%intilisation du poids
W=randn(2+1,1)% +1 pour le biais
pas=0.1
for j=1:5
for i=1:size(X,1)
    erreur=10;
    while(erreur> 0.001)        
    O = [X(i,:) 1]*W
    if(O>0)
        O=1;
    else
        O=0;
    end 
    erreur=C(i)-O
    W=W+2*pas*erreur*[X(i,:) 1]';
    end
end
end
%6. Tracer la limite de decision (plotpc)
plotpc(W(1:2)',W(3))


