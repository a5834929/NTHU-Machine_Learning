classdef SMOClassifier < handle
   
    properties
        alpha, bias;
        sv, svIndex;
    end
    
    methods
        function smoClassifierObj = SMOClassifier(alpha, bias, sv, svIndex)
            smoClassifierObj.alpha = alpha;
            smoClassifierObj.bias = bias;
            smoClassifierObj.sv = sv;
            smoClassifierObj.svIndex = svIndex;
        end
        
        function predictedLabel = predict(obj, X)
            kfun = @obj.rbf_kernel;
            f = (feval(kfun,obj.sv,X,32)'*obj.alpha) + obj.bias;
            predictedLabel = sign(f);
            predictedLabel(predictedLabel==0) = 1;
        end
    end
    
    methods (Static)
        function smoClassifierObj = train(X, y)
            K = model.classify.SMOClassifier.rbf_kernel(X,X,32);
            alpha = zeros(size(y,1),1);
            grad = ones(size(y,1),1);
            bias = 0;
            C = 2*ones(size(y,1),1); 
            A = zeros(size(y,1),1);
            B = zeros(size(y,1),1);
            A(y==-1) = -C(y==-1);  
            B(y==1)  = C(y==1);
            index = 1:size(y,1);   
            
            while 1
                Iup     = y.*alpha < B;
                Idown   = y.*alpha > A;
                
                [vali, i] = max(y(Iup).*grad(Iup));
                tmp = index(Iup);
                i = tmp(i);
                [valj, j] = min(y(Idown).*grad(Idown));
                tmp = index(Idown);
                j = tmp(j);
                
                if vali-valj <= 1e-4
                    bias = (vali+valj)/2;
                    svIndex = index(alpha~=0);
                    sv = X(svIndex, :);
                    alpha = alpha(svIndex).*y(svIndex);
                    smoClassifierObj = model.classify.SMOClassifier(alpha, bias, sv, svIndex);
                    return;
                end
                
                lam(1) = B(i)-y(i)*alpha(i);
                lam(2) = y(j)*alpha(j)-A(j);
                lam(3) = (y(i)*grad(i)-y(j)*grad(j))/(K(i,i)+K(j,j)-2*K(i,j));
                lambda = min(lam);
                
                grad = grad-lambda*y.*(K(i,:)-K(j,:))';
                
                alpha(i) = alpha(i)+y(i)*lambda;
                alpha(j) = alpha(j)-y(j)*lambda;
            end
        end
        
        function kval = rbf_kernel(u,v,n)
            kval = exp(-(1/(2*n^2))*(repmat(sqrt(sum(u.^2,2).^2),1,size(v,1))...
            -2*(u*v')+repmat(sqrt(sum(v.^2,2)'.^2),size(u,1),1)));
        end
    end
end   

