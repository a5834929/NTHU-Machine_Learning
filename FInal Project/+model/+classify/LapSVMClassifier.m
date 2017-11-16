classdef LapSVMClassifier < handle
    
    properties
        alpha, data, rbf, transMat;
    end
    
    methods
        function LapSVMClassifierObj = LapSVMClassifier(alpha, data, rbf, transMat)
            LapSVMClassifierObj.alpha = alpha;
            LapSVMClassifierObj.data = data;
            LapSVMClassifierObj.rbf = rbf;
            LapSVMClassifierObj.transMat = transMat;
        end
        
        function predictedLabel = predict(obj, X)
            X = X*obj.transMat;
            kfun = @obj.rbf_kernel;
            f = (feval(kfun,X,obj.data,obj.rbf)*obj.alpha);
            predictedLabel = sign(f);
        end
    end
    
    methods (Static)
        %
        function LapSVMClassifierObj = train(X,y)
            %feature selection
            transMat = model.classify.LapSVMClassifier.PCA(X,3);
            X = X*transMat;
            
            %denoise
            %noise = [5 18 34 44 45];
            %X(noise,:) = [];
            %y(noise) = [];
            
            N = size(X,1);
            sigma = 0.3;
            rbf = 0.4;
            lambda = 0.6;
            mu = 0.5;
            
            l = nnz(y);
            S = squareform(pdist(X));
            S = exp(-(S.^2/sigma^2));
            D = diag(sum(S));
            L = D-S;
            I = eye(l);
            R = diag(nonzeros(y));
            J = I;
            J(l,N)=0;
            
            K = model.classify.LapSVMClassifier.rbf_kernel(X,X,rbf);
            Q = R*J*inv(2*lambda*eye(N)+2*mu*L*K)'*K*J'*R;
           
            cvx_begin quiet
                variable b(l) nonnegative
                maximize(sum(b)-(b'*Q*b)/2);
                subject to
                    b'*y(1:l) == 0;
                    b <= 1/l;
            cvx_end
            
            alpha = inv(2*lambda*eye(N)+2*mu*L*K)*J'*R*b;
            LapSVMClassifierObj = model.classify.LapSVMClassifier(alpha, X, rbf, transMat);
        end
        %{
        
        function LapSVMClassifierObj = train(X,y,sigma,rbf,lambda,mu)
            transMat = model.classify.LapSVMClassifier.PCA(X,3);
            X = X*transMat;
            N = size(X,1);

            l = nnz(y);
            S = squareform(pdist(X));
            S = exp(-(S.^2/sigma^2));
            D = diag(sum(S));
            L = D-S;
            I = eye(l);
            R = diag(nonzeros(y));
            J = I;
            J(l,N)=0;
            
            K = model.classify.LapSVMClassifier.rbf_kernel(X,X,rbf);
            Q = R*J*inv(2*lambda*eye(N)+2*mu*L*K)'*K*J'*R;
           
            cvx_begin quiet
                variable b(l) nonnegative
                maximize(sum(b)-(b'*Q*b)/2);
                subject to
                    b'*y(1:l) == 0;
                    b <= 1/l;
            cvx_end
            
            alpha = inv(2*lambda*eye(N)+2*mu*L*K)*J'*R*b;
            LapSVMClassifierObj = model.classify.LapSVMClassifier(alpha, X, rbf, transMat);
        end
        %}
        
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

