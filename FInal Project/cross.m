import model.classify.LapRLSClassifier
import model.classify.LapSVMClassifier

load X.mat;
load y.mat;
load Xtest.mat;
load ytest.mat;
noise = [5 18 34 44 45];
X(noise,:) = [];
y(noise) = [];
cnt = 1;
for mu = 0.5:0.1:0.6
    for lambda = 0.5:0.1:0.6
        for sigma = 0.3:0.05:0.4
            for rbf = 0.3:0.05:0.4
                err = 0;
                for i=1:45
                    x = X;
                    yy = y;
                    x(i,:)= [];
                    yy(i) = [];
                    %myClassifier = model.classify.LapRLSClassifier.train(x,yy,sigma,rbf,lambda,mu);
                    myClassifier = model.classify.LapSVMClassifier.train(x,yy,sigma,rbf,lambda,mu);
                    label = myClassifier.predict(X(i,:));
                    err = err+(label~=y(i));
                end
                acc1 = 1-err/45;
                if acc1>0.88
                    %myClassifier = model.classify.LapRLSClassifier.train(X,y,sigma,rbf,lambda,mu);
                    myClassifier = model.classify.LapSVMClassifier.train(x,yy,sigma,rbf,lambda,mu);
                    label = myClassifier.predict(Xtest);
                    acc2 = 1-sum(label~=ytest)/50;
                    result(cnt,1:6) = [sigma rbf lambda mu acc1 acc2];
                    fprintf('sigma %g rbf %g lambda %g mu %g\n    validation acc %g testing acc %g\n', result(cnt,:));
                    cnt=cnt+1;
                end
            end
        end
    end
end

fid = fopen('LapSVM.txt', 'w');

for i=1:cnt-1
    fprintf(fid, 'sigma %g rbf %g lambda %g mu %g\n    validation acc %g testing acc %g\n', result(i,:));
end
