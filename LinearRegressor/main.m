import model.regressor.LinearRegressor
import model.regressor.LinearRegressorLocalWeight

close all;
clear;

load('X.dat');
load('y.dat');

%---------Linear Regressor-------------------------------------------------
linearRegressor = LinearRegressor.train(X, y);
linearValue = linearRegressor.predict(X);

scatter(X, y, 'g');
title('Linear Regressor');
hold on;
plot(X, linearValue, 'r');
hold off;
axis tight;

%---------Linear Regressor Locally-Weighted--------------------------------
tau = [0.1, 1, 10, 100];
newX = min(X):0.5:max(X);
for i=1:4
    
cfg = containers.Map('tau', tau(i));
localWeightLinearRegressor = LinearRegressorLocalWeight.train(X, y);
localWeightValue = localWeightLinearRegressor.predict(newX, cfg);

figure;
scatter(X, y, 'g');
title(['Locally Weighted Linear Regressor (\tau = ' num2str(tau(i)) ')']);
hold on;
scatter(newX, localWeightValue, 'r');
hold off;
axis tight;

end

