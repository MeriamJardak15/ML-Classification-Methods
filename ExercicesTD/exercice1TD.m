close all
clear all
clc

%1. Créer le vecteur des poids nécessaires.
w1=1;
w2=1;
poids=[w1;w2];
%2. Ajouter un biais.
b=-5;
%3. Choisir la fonction à tester.
f="tansig";
%observations
d=[3;4];
%4. Créer et former un perceptron
somme=d'*poids+b;
%5. Évaluer à la fin la fonction de sortie de ce perceptron. (feval)
sortie=feval(f,somme);

%1. Créer une grille sur un carré entre -10et 10 par exemple. (meshgrid)
[abs,ord]=meshgrid(-10:0.5:10);
%2. Évaluer la totalité des points sur ce perceptron.
X=[abs(:) ord(:)];
sortiegrille=feval(f,X*poids+b);
sortiegrille=reshape(sortiegrille,length(abs),length(ord));
%3. Tracer la sortie en 3D. (plot3)
figure;
plot3(abs,ord,sortiegrille)


