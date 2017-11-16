classdef SoftMarginLinearClassifier < handle
    
    properties
        w, b, slack;
    end
    
    methods
        function softMarginClassifierObj = SoftMarginLinearClassifier(w, b, slack)
            softMarginClassifierObj.w = w;
            softMarginClassifierObj.b = b;
            softMarginClassifierObj.slack = slack;
        end
        function predictedLabel = predict(obj, X)
            featureX = X;
            predictedLabel = sign(obj.w'*featureX'-obj.b);
            size(predictedLabel)
        end
    end
    
    methods (Static)
        function softMarginClassifierObj = train(X, y)
            featureX = model.classify.SoftMarginLinearClassifier.lift(X);
            n = size(featureX, 1);
            d = size(featureX, 2);
            Q = eye(d);
            Q(d+n+1, d+n+1) = 0;
            C = 1;
            c = zeros(d+1, 1);
            c = C*cat(1, c, ones(n, 1));
            for i=1:n
                G(i, :) = -y(i)*featureX(i, :);
            end
            G(1:n, d+1) = y;
            G(1:n, d+2:d+n+1) = -eye(n);
            G(n+1:2*n, d+2:d+n+1) = -eye(n);
            h = cat(1, -1*ones(n, 1), zeros(n, 1));
            
            cvx_begin
                variable x(d+n+1)
                minimize(x'*Q*x + c'*x)
                subject to
                    G*x <= h
            cvx_end
            
            w = x(1:d);
            b = x(d+1);
            slack = (d+2:d+n+1);
            softMarginClassifierObj = model.classify.SoftMarginLinearClassifier(w, b, slack);
        end
        
        function featureX = lift(X)
            featureX = X;
            %featureX = [X(:, 1).^3, X(:, 2).^3, ones(size(X, 1), 1), 0.001*X(:, 1).^2.*X(:, 2), sqrt(3)*X(:, 1).*X(:, 2).^2, 100*X(:, 2).^2, sqrt(3)*X(:, 2), 0.0001*X(:, 1).^2, sqrt(3)*X(:, 1), sqrt(3)*X(:, 1), sqrt(6)*X(:, 1).*X(:, 2)];
        end
    end
    
end

