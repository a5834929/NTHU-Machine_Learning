classdef LapRLSClassifier < handle
  
    properties
        alpha, sigma, data;
    end
    
    methods
        function LapRLSClassifierObj = LapRLSClassifier(alpha, sigma, data)
            LapRLSClassifierObj.alpha = alpha;
            LapRLSClassifierObj.sigma = sigma;
            LapRLSClassifierObj.data = data;
        end
        
        function predictedLabel = predict(obj, X)
            kfun = @obj.rbf_kernel;
            f = (feval(kfun,X,obj.data,obj.sigma)*obj.alpha);
            predictedLabel = sign(f);
        end
    end
    
    methods (Static)
        function LapRLSClassifierObj = train(X,y)
            sizeX = size(X,1);
            s1 = pdist(X(:,1));
            s2 = pdist(X(:,2));
            S = squareform(s1)+squareform(s2);
            sigma = 1;
            S = exp(-(S.^2/sigma^2));
            D = diag(sum(S));
            L = D-S;
            J = diag(y~=0);
            K = model.classify.LapRLSClassifier.rbf_kernel(X,X,sigma);
            lambda = 3;
            mu = 0.1;
            alpha = inv(J*K+mu*L*K+lambda*eye(sizeX))*J*y;
            LapRLSClassifierObj = model.classify.LapRLSClassifier(alpha, sigma, X);
        end
        
        function kval = rbf_kernel(u,v,n)
            kval = exp(-(1/(2*n^2))*(repmat(sqrt(sum(u.^2,2).^2),1,size(v,1))...
            -2*(u*v')+repmat(sqrt(sum(v.^2,2)'.^2),size(u,1),1)));
        end
    end
    
end

