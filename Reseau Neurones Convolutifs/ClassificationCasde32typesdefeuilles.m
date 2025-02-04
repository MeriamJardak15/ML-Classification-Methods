close all
clear all
clc

%Chemin d'accès et Chargement du données
digitDatasetPath='C:\Users\LENOVO\OneDrive\Bureau\Imagors';
imds=imageDatastore(digitDatasetPath,'IncludeSubfolders',true,'LabelSource','foldernames');

%Affichage des données
perm=randperm(4955,20); % generer d'une manière aléatoire 20 entiers entre 1 et 4955
for i=1:20
   subplot(4,5,i);
   imshow(imds.Files{perm(i)});
   hold on
end
 
%Nombre des images pour chaque catégories
labelCount=countEachLabel(imds)
%Taille des images
img=readimage(imds,500);
size(img)

%Division des données
[trainDigitData,valDigitData]=splitEachLabel(imds,0.75,0.25,'randomized');


%Définition de l'architecture du réseau
layers = [
imageInputLayer([175 175 3])
  convolution2dLayer(3,10,'Padding','same')
  batchNormalizationLayer
  reluLayer
  maxPooling2dLayer(2,'Stride',2)
  convolution2dLayer(3,18,'Padding','same')
  batchNormalizationLayer
  reluLayer
  maxPooling2dLayer(2,'Stride',2)
  convolution2dLayer(3,26,'Padding','same')
  batchNormalizationLayer
  reluLayer
  maxPooling2dLayer(2,'Stride',2)
  convolution2dLayer(3,34,'Padding','same')
  batchNormalizationLayer
  reluLayer  
  fullyConnectedLayer(32)
  softmaxLayer
  classificationLayer];

%Option d'apprentissage(Nombre d'époques)
options = trainingOptions('sgdm', ...
 'InitialLearnRate',0.01, ...
 'MaxEpochs',5, ...
 'Shuffle','every-epoch', ...
 'ValidationData',valDigitData, ...
 'ValidationFrequency',30, ...
 'Verbose',false, ...
 'Plots','training-progress');
%Apprentissage du réseau
net=trainNetwork(trainDigitData,layers,options);

%Performance du réseau
Ypred=classify(net,valDigitData);
Yvalidation=valDigitData.Labels; 
accurary=sum(Ypred==Yvalidation)/length(Ypred)
