classdef SpectralClustering < handle
    methods (Static)
        function clusterObj = cluster(X, k, cfg)
            S = eye(size(X,1));
            dPair = pdist(X);
            dPair = squareform(dPair);
    
            similarity = cfg('similarity');
            switch similarity
                case 'eNN'
                    epsilon = 12;
                    [~,sortIndex] = sort(dPair, 2, 'ascend');
                    minDist = sortIndex(:,1:epsilon+1);
                    for i=1:size(X,1)
                        S(i,minDist(i,2:epsilon+1)) = dPair(i,minDist(i,2:epsilon+1));
                        S(minDist(i,2:epsilon+1),i) = dPair(minDist(i,2:epsilon+1),i);
                    end
                case 'eBall'
                    epsilon = 1.2;
                    S(dPair<epsilon) = dPair(dPair<epsilon);
                case 'Gaussian'
                    sigma = 0.56;
                    S = exp(-(dPair.^2/sigma^2));
            end
            
            D = diag(sum(S));
            L = D-S;     
            opt = struct('issym', true, 'isreal', true);
            [V, ~] = eigs(L, D, k, 'SM', opt);
            
            clusterObj = model.clustering.KmeansClustering.cluster(V, k);
        end
    end
end

