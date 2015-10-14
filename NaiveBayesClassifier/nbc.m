%{
Naive Bayes Classifier
-----------------------

P(c/w)=argmax{P(w/c) * P(c)}

P(cj) is the Prior.
P(cj)= # of docs in class cj/(total # of docs in class cj)

P(wi/cj) is the likelihood for word i given class j.
P(wi/cj)=(count of wi occuring in cj)/Sum of all words in cj.

In this code:
p(cj) is represented as varaible "p_c".
P(wi/cj) is represented as variable "pwc".

count of wi occuring in cj is represented as "count_w_c".
Sum of all words in cj is represented as "tot_count_w_w".



%}

traintotaldocs=11269;
testtotaldocs=7505;
traininput=dlmread('train.data');
trainl=dlmread('train.label');
testinput=dlmread('test.data');
testl=dlmread('test.label');
vocabulary=textread('vocabulary.txt','%s');

beta=1/61188;
alpha=1+beta;


% Prior of each class caluclations starts


p_c=zeros(20,1);
ycount=zeros(20,1);
for i=1:traintotaldocs
  ycount(trainl(i))=ycount(trainl(i))+1;
end
p_c=ycount./traintotaldocs; % p(c) obtained. 

% Prior of each class caluclations ends


%Likelihood caluclations starts here

count_w_c=zeros(20,61188);
counter = 0;

 
for i=1:1467345
  count_w_c(trainl(traininput(i,1)),traininput(i,2))=count_w_c(trainl(traininput(i,1)),traininput(i,2))+traininput(i,3);
end  

tot_count_w_w=sum(count_w_c,2)+((alpha-1)*61188);
pwc=zeros(20,61188);
for i=1:20
  pwc(i,:)=(count_w_c(i,:)+(alpha-1))./tot_count_w_w(i);    %% p(w/c) obtained. 
end



%Likelihood caluclations ends here

%caluclating ynew start here
testdata=zeros(7505,61188);

for i=1:967874
  testdata(testinput(i,1),testinput(i,2))= testdata(testinput(i,1),testinput(i,2))+testinput(i,3);
end

tempmat=log2(pwc);
pwct=transpose(tempmat);
pos=testdata*pwct;
priormain=log2(p_c);

post=transpose(pos);
newprior=zeros(20,7505);
for i=1:7505
newprior(:,i)=priormain(:,1);
end

ynew=newprior+post;
%ynew cluclation ends here

%Testing the predicted labels starts here

[M,index] = max(ynew);
index=transpose(index);
for i=1:7505
    if(testl(i,1)==index(i,1))
        counter=counter+1;
    end
end

Accuracy=(counter/7505)*100;

%Testing the predicted labels starts here

%Caluclating the confusion matrix starts here

confusion_matrix=zeros(20,20);

for i=1:7505
  confusion_matrix(testl(i,1),index(i,1))= confusion_matrix(testl(i,1),index(i,1))+1;
end

%Caluclating the confusion matrix starts here



%Proposed method and top 100 vocab list

xp=zeros(1,61188);
for i=1:61188
  xp(i)=sum(pwc(:,i).*p_c);
end

ex=-xp.*log2(xp)-(1-xp).*log2(1-xp);

ixy=(ex)';
[temp,index1]=sort(ixy,'ascend');

vocabulary{index1(1:100)}