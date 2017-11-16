%%% the linear regressor predicts the value of y by linear approach

classdef LinearRegressor < handle
   properties
      w;
   end
   
   methods
       function linearRegressorObj = LinearRegressor(w)  % constructor
           linearRegressorObj.w = w;
       end
       function predictedValue = predict(obj, X)
           predictedValue = obj.w'*[ones(1, length(X));X'];
       end
   end
   
   methods (Static)
      function linearRegressorObj = train(X, y)
        A = [size(y, 1) sum(X);sum(X) X'*X];
        Y = [sum(y);X'*y];
        theta = A\Y;
        linearRegressorObj = model.regressor.LinearRegressor(theta);
      end
   end
end