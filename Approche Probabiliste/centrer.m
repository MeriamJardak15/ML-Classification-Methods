function Xns= centrer(Xns)
n=size(Xns(1,:),2);
for i=1:n
    Xns(:,i)=(Xns(:,i)-mean(Xns(:,i)))/std(Xns(:,i));
end
end

