import model.classify.RLSClassifier
import model.classify.LapRLSClassifier

close all;
clear;

X = load('X.mat');
X = X.X;
y = load('y.mat');
y = y.y;
sol = load('y_sol.mat');
sol = sol.y_sol;

%myClassifier = RLSClassifier.train(X,y);
myClassifier = LapRLSClassifier.train(X,y);
predicted = myClassifier.predict(X);

%{
len=1;
for i=-3:0.02:3
    for j=-4:0.02:8
        XX(len,1) = i;
        XX(len,2) = j;
        len=len+1;
    end
end
%}
%XX = [-2 -2;2.5 4;1 2;-1 0];
%predicted = myClassifier.predict(XX);

hold on;
%scatter(X(:,1), X(:,2), 'k');
scatter(X(predicted==1,1), X(predicted==1,2), 'b');
scatter(X(predicted==-1,1),X(predicted==-1,2));
%scatter(X(sol==1,1), X(sol==1,2), 'c.');
%scatter(X(sol==-1,1), X(sol==-1,2), 'g.');
title('RLS Classifier');

error = sum(predicted~=sol)

