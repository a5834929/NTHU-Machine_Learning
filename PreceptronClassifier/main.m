close all;
clear;

testX = load('testing/X.dat');
testy = load('testing/y.dat');
N = [10 100 1000];
for i=1:3
    X = load(['training/' num2str(N(i)) '/X.dat']);
    y = load(['training/' num2str(N(i)) '/y.dat']);
    lift = 1;
    
    perceptronClassifier = PerceptronClassifier.train(X, y, lift);
    generalizationError = perceptronClassifier.GeneralizationError(testX, testy, size(testX, 1), lift);
    consistencyBound = perceptronClassifier.ConsistencyBound(N(i), lift);
    
    err(i) = generalizationError;
    bound(i) = consistencyBound;

end

plot(N, err, 'r');hold on;
plot(N, bound, 'b');hold off;
legend('Generalization Error', 'Consistency Bound');
title(['Perceptron Classifier (\Phi ' num2str(lift) ')'], 'FontSize', 20);

