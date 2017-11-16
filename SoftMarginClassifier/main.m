import model.classify.SoftMarginLinearClassifier
import model.classify.SMOClassifier

close
clear

X = load('X.dat');
y = load('y.dat');


myClassifier = SoftMarginLinearClassifier.train(X,y);
label = myClassifier.predict(X);
w = myClassifier.w;
b = myClassifier.b;


%myClassifier = SMOClassifier.train(X,y);
alpha = myClassifier.alpha;
bias = myClassifier.bias;

n1 = min(X(:,1));
n2 = max(X(:,1));
m2 = max(X(:,2));

i = 1;
%{
while n1<=n2
   m1 = min(X(:,2));
   while m1<=m2
       X_(i,1) = n1;
       X_(i,2) = m1;
       i = i+1;
       m1 = m1+0.5;
   end
   n1 = n1+0.5;
end
%}
%{
ii = 1;
for i=-100:0.5:80
   for j=-150:0.5:150
       X_(ii, 1)=i;
       X_(ii, 2)=j;
       ii = ii+1;
   end
end

label = myClassifier.predict(X_);
%}
lb = myClassifier.predict(X);

%featureX = [ones(size(X, 1), 1), X(:,1), X(:,2), X(:,1).^2, X(:,2).^2, X(:,1).*X(:,2)];
%K = featureX*featureX';
%[alpha, bias] = smo(K, y', 0.1, 0.1);
%w(1) = sum(alpha'.*y.*featureX(:,1));
%w(2) = sum(alpha'.*y.*featureX(:,2));
%label = sign(featureX(:,1)*w(1)+featureX(:,2)*w(2)+bias);

%{
hold on;
scatter(X(y==1,1),X(y==1,2),'g');
scatter(X(y==-1,1),X(y==-1,2), 'b');
scatter(X(lb'~=y,1),X(lb'~=y,2), 'r.');
[sx, xin] = sort(X(:,1));
%}

%plot(sx, (b-sx.*w(1))/w(2), 'r');

%{
scatter(X_(label==1,1),X_(label==1,2),'g');
scatter(X_(label==-1,1),X_(label==-1,2),'b');
scatter(X(y~=lb,1), X(y~=lb,2), 'r.');
err = sum(y~=lb)
title(['C=1    \sigma =32    error=' num2str(err) '    #SV=' num2str(size(alpha,1))]);
%}
hold on;
scatter(X(y==1,1),X(y==1,2),'g');
scatter(X(y==-1,1),X(y==-1,2), 'b');
scatter(X(y~=lb,1), X(y~=lb,2), 'r.');
err = sum(y~=lb)


%[sx, ind] = sort(X(:,1));
%plot(sx, X(ind,1)*w(1)+X(ind,2)*w(2)+bias, 'r');


%error = sum(y~=label)
%hold off;
