import model.clustering.KmeansClustering
import model.clustering.SpectralClustering

close all
clear

data = load('data.dat');

%-----------------------KmeansClustering-----------------------------------

k=6;
myCluster = KmeansClustering.cluster(data, k);
y = myCluster.Y;
center = myCluster.center;
color = hsv(k);

hold on;
for i=1:k
    scatter(data(y(:,i)==1,1),data(y(:,i)==1,2),...
            40,ones(sum(y(:,i)==1),1)*color(i, :),'.');
end
scatter(center(:,1),center(:,2),'bO');

%-----------------------SpectralClustering---------------------------------
%{
k=2;
similarity = {'eNN' 'eBall' 'Gaussian'};
cfg = containers.Map('similarity', similarity{3});
myCluster = SpectralClustering.cluster(data, k, cfg);
y = myCluster.Y;
center = zeros(k,2);
color = hsv(k);

hold on;
title(['K = ' num2str(k)]);
for i=1:k
    scatter(data(y(:,i)==1,1),data(y(:,i)==1,2),...
            40,ones(sum(y(:,i)==1),1)*color(i, :),'.');
    center(i,1) = mean(data(y(:,i)==1,1));
    center(i,2) = mean(data(y(:,i)==1,2));
    scatter(center(i,1),center(i,2),'ko');
end
%}
separation = 2*sum(pdist(center).^2);
cohesion = 0;

for i=1:k
    sizeG = sum(y(:,i)==1);
    centeri = repmat(center(i,:), sizeG, 1);
    cohesion = cohesion + sum(sum((data(y(:,i)==1, :)-centeri).^2))/sizeG;
end

quality = separation/cohesion
title(['K = ' num2str(k) '    Quality = ' num2str(quality)]);

