load X.mat;
load y.mat;

close all;
hold on;

%{
for i=1:50
    if y(i)==1
        scatter3(X(i,1), X(i,2), X(i,3), 'g');
    else
        scatter3(X(i,1), X(i,2), X(i,3), 'b');
    end
end
h = 1:50;
value = num2str(h');
text(X(1:50,1), X(1:50,2), X(1:50,3), value,'horizontal','left', 'vertical','bottom');
%}

%scatter(X(y==0,1), X(y==0,3),'y.');
%scatter3(X(y==1,1), X(y==1,2), X(y==1,3),'g');
scatter3(X(noise,1), X(noise,2), X(noise,3),'r');
scatter3(X(y==1,1), X(y==1,2), X(y==1,3),'g.');
%scatter3(X(y==-1,1), X(y==-1,2), X(y==-1,3),'b');
scatter3(X(y==-1,1), X(y==-1,2), X(y==-1,3),'b.');
noise = [5 18 34 44 45];

xlabel('Attribute 1');
ylabel('Attribute 2');
zlabel('Attribute 3');
%box on;