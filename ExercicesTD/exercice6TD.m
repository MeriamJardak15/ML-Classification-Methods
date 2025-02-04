close all
clear all
clc
%6.1. Charger et examiner les données
digitDatasetPath=fullfile(matlabroot,'toolbox/nnet/nndemos/nndatasets/DigitDataset');
imds=imageDatastore(digitDatasetPath,'IncludeSubfolders',true,'LabelSource','foldernames');
figure;
imshow(imds.Files{4822})
labelCount=countEachLabel(imds)
img=readimage(imds,500);
size(img)
%6.2. Préciser les ensembles d'apprentissage et de validation
[trainDigitData,valDigitData]=splitEachLabel(imds,750,'randomized')
%6.3. Définir l'architecture du réseau
layers = [
 imageInputLayer([28 28 1])
 convolution2dLayer(3,8,'Padding','same')
 batchNormalizationLayer
 reluLayer
 maxPooling2dLayer(2,'Stride',2)
 fullyConnectedLayer(10)
 softmaxLayer
 classificationLayer];
%6.4. Préciser les options d’apprentissage
options = trainingOptions('sgdm', ...
 'InitialLearnRate',0.01, ...
 'MaxEpochs',4, ...
 'Shuffle','every-epoch', ...
 'ValidationData',valDigitData, ...
 'ValidationFrequency',30, ...
 'Verbose',false, ...
 'Plots','training-progress');
%6.5. Apprendre le réseau avec les données d'apprentissage
net=trainNetwork(trainDigitData,layers,options);
%6.6. Classer les images de validation images et calculer la précision
Ypred=classify(net,valDigitData);
Yvalidation=valDigitData.Labels;
accurary=sum(Ypred==Yvalidation)/length(Ypred)

