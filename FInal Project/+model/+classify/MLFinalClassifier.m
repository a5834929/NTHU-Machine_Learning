classdef MLFinalClassifier < handle
    properties
        alpha, data, rbf, transMat;
    end
    
    methods
        function MLFinalClassifierObj = MLFinalClassifier(alpha, data, rbf, transMat)
            MLFinalClassifierObj.alpha = alpha;
            MLFinalClassifierObj.data = data;
            MLFinalClassifierObj.rbf = rbf;
            MLFinalClassifierObj.transMat = transMat;
        end
        
        function predictedLabel = predict(obj, X)
            X = X*obj.transMat;
            kfun = @obj.rbf_kernel;
            f = (feval(kfun,X,obj.data,obj.rbf)*obj.alpha);
            predictedLabel = sign(f);
        end
    end
    
    methods (Static)
        function MLFinalClassifierObj = train(X,y)
            %dimensionality reduction
            transMat = model.classify.MLFinalClassifier.PCA(X,3);
            X = X*transMat;
            
            %denoise
            noise = [5 18 34 44 45];
            X(noise,:) = [];
            y(noise) = [];
            
            N = size(X,1);
            sigma = 0.3;
            rbf = 0.4;
            lambda = 0.575;
            mu = 0.51;
            
            S = squareform(pdist(X));
            S = exp(-(S.^2/sigma^2));
            D = diag(sum(S));
            L = D-S;
            J = diag(y~=0);
            K = model.classify.MLFinalClassifier.rbf_kernel(X,X,rbf);
            
            alpha = inv(J*K+mu*L*K+lambda*eye(N))*J*y;
            MLFinalClassifierObj = model.classify.MLFinalClassifier(alpha, X, rbf, transMat);
        end

        function transMat = PCA(X,n)
            Xmean = mean(X);
            X = X - repmat(Xmean, size(X,1), 1);
            covMat = cov(X);
            [V, D] = eig(covMat);
            [~, idx] = sort(diag(D), 'ascend');
            transMat = V(:, idx(1:n));
        end
        
        function kval = rbf_kernel(u,v,n)
            kval = exp(-(1/(2*n^2))*(repmat(sqrt(sum(u.^2,2).^2),1,size(v,1))...
            -2*(u*v')+repmat(sqrt(sum(v.^2,2)'.^2),size(u,1),1)));
        end
    end
    
end

