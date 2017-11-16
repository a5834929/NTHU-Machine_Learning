classdef KmeansClustering < handle
    properties
        Y, center;
    end
    
    methods
        function clusterObj = KmeansClustering(Y, center)
            clusterObj.Y = Y;
            clusterObj.center = center;
        end
    end
    
    methods (Static)
        function clusterObj = cluster(X, k)
            y = zeros(size(X,1), 1);
            r = randi([1 size(X,1)],1,1);
            C = X(r,:);
            sizeC = size(C,1);
            
            while sizeC<k
                for i=1:size(X,1)
                    xi = repmat(X(i,:), sizeC, 1);
                    di = sum((xi-C).^2, 2);
                    minD = min(di);
                    weight(i) = minD;
                end
                r = randsample(1:size(X,1), 1, true, weight);
                C(end+1,:) = X(r,:);
                sizeC = size(C,1);
            end
           
            while 1
                for i=1:size(X,1)
                    xi = repmat(X(i,:), k, 1);
                    di = sum((xi-C).^2, 2);
                    [~, minC] = min(di);
                    y(i) = minC;
                end
                
                oldC = C;
                for i=1:k
                    C(i,:) = mean(X(y==i, :));
                end
                
                if norm(oldC-C)<eps
                    for i=1:k
                        Y(:,i) = (y==i);
                    end
                    clusterObj = model.clustering.KmeansClustering(Y, oldC);
                    break;
                end
            end
        end
    end
    
end

