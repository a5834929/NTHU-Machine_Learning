%%% the linear regressor predicts the value of y by linear approach

classdef LinearRegressorLocalWeight < model.regressor.LinearRegressor
   properties
      X;
      y;
      tau;
   end
   
   methods
       function localWeightObj = LinearRegressorLocalWeight(X, y)  % constructor
           localWeightObj = localWeightObj@model.regressor.LinearRegressor(1);
           localWeightObj.X = X;
           localWeightObj.y = y;
       end
       
       function weight = CalculateWeight(~, X, tau, newX)
           weight = exp(-((X-newX).^2)/(2*(tau^2)));
       end
       
       function predictedValue = predict(obj, X, cfg)
           obj.tau = cfg('tau');
           predictedValue = zeros(length(X), 1);
           for i=1:length(X)
                weight = obj.CalculateWeight(obj.X, obj.tau, X(i));
                A = [sum(weight) weight'*obj.X;weight'*obj.X sum(weight.*obj.X.*obj.X)];
                Y = [weight'*obj.y;sum(weight.*obj.X.*obj.y)];
                obj.w = A\Y;
                predictedValue(i) = [1 X(i)]*obj.w;
           end
       end
   end
   
   methods (Static)
      function localWeightObj = train(X, y)
        localWeightObj = model.regressor.LinearRegressorLocalWeight(X, y);
      end
   end
end