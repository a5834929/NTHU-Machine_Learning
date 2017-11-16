
classdef PerceptronClassifier < handle
    properties
        w;
        b;
    end
   
    methods
        function perceptronClassifierObj = PerceptronClassifier(w, b)
          perceptronClassifierObj.w = w;
          perceptronClassifierObj.b = b;
        end
  
        function generalErr = GeneralizationError(obj, X, r, N, lift)
            liftX = obj.phi(X, lift);
            predictY = liftX*obj.w + obj.b;
            y = sign(predictY);
            generalErr = sum(y~=r)/N;
        end
        
        function bound = ConsistencyBound(obj, N, lift)
            if(lift==10), Rh = 1/10;
            else         Rh = 0;
            end
            vc = obj.VCDimension(lift);
            bound = Rh+2*sqrt((32/N)*(vc*log10(N*exp(1)/vc)+log10(40)));
        end
        
        function vc = VCDimension(~, lift)
            vc = lift+1;
        end
    end
   
    methods (Static)
        function perceptronClassifierObj = train(X, r, lift)
            featureX = PerceptronClassifier.phi(X, lift);
            W = randn(lift, 1);
            B = randn;
            err = 1e-10;
            for i=0:10000
                preW = W;
                preB = B;
                predictY = featureX*W + B;
                y = sign(predictY);
                eta = norm((r-y)'*featureX)/3;
                W = preW + eta*((r-y)'*featureX)';
                B = preB + eta*sum(r-y);
                if norm(W-preW)<err
                    break;
                end
            end
            perceptronClassifierObj = PerceptronClassifier(W, B);
        end
        
        function featureX = phi(X, lift)
            featureX = X;
            while lift>1
                featureX = [X.^lift featureX];
                lift = lift-1;
            end
        end
    end
end