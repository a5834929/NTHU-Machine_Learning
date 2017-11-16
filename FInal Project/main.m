import model.classify.LapRLSClassifier
import model.classify.LapSVMClassifier

close all;
clear;

load X.mat;
load y.mat;
load Xtest.mat;
load ytest.mat;

%myClassifier = model.classify.LapRLSClassifier.train(X, y);
%myClassifier = model.classify.LapSVMClassifier.train(X, y);
myClassifier = model.classify.MLFinalClassifier.train(X, y);
label = myClassifier.predict(Xtest);

hold on;
scatter3(Xtest(ytest==1,1), Xtest(ytest==1,2), Xtest(ytest==1,3), 'g');
scatter3(Xtest(ytest==-1,1), Xtest(ytest==-1,2), Xtest(ytest==-1,3), 'b');
scatter3(Xtest(ytest~=label,1), Xtest(ytest~=label,2), Xtest(ytest~=label,3), 'rx');
title('Denoised');

err = sum(label~=ytest)
accuracy = 1-err/length(ytest)

