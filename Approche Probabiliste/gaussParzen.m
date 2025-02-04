function z=gaussParzen(data,appr,sig)

% data: point(s) x
% appr: données d'apprentissage
% sig: std du noyau de parzen
Sigma=sig^2*eye(size(data,1));
% Sigma=sig^2*eye(size(appr,2));
N=size(appr,1);
%z=zeros(size(data,1),1);
z=zeros(size(data,2),1);

for i=1:N
    z=z+mvnpdf(data',appr(i,:),Sigma);% à modifier
end

z=z/N;

