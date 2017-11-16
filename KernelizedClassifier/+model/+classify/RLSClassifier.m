classdef RLSClassifier < handle
  
    properties
        alpha, sigma, data;
    end
    
    methods
        function RLSClassifierObj = RLSClassifier(alpha, sigma, data)
            RLSClassifierObj.alpha = alpha;
            RLSClassifierObj.sigma = sigma;
            RLSClassifierObj.data = data;
        end
        
        function predictedLabel = predict(obj, X)
            kfun = @obj.rbf_kernel;
            f = (feval(kfun,X,obj.data,obj.sigma)*obj.alpha);
            predictedLabel = sign(f);
        end
    end
    
    methods (Static)
        function RLSClassifierObj = train(X,y)
            sizeX = size(X,1);
            lambda = 3;
            sigma = 1;
            K = model.classify.RLSClassifier.rbf_kernel(X,X,sigma);
            alpha = inv(K+lambda*eye(sizeX))*y;
            RLSClassifierObj = model.classify.RLSClassifier(alpha, sigma, X);
        end
        
        function kval = rbf_kernel(u,v,n)
            kval = exp(-(1/(2*n^2))*(repmat(sqrt(sum(u.^2,2).^2),1,size(v,1))...
            -2*(u*v')+repmat(sqrt(sum(v.^2,2)'.^2),size(u,1),1)));
        end
    end
    
end

