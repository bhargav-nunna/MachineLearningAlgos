function acc=prj2(input)

beta=1/61188;
traintotaldocs=11269;
testtotaldocs=7505;
vocablist=61188;

totaldocsperclass=zeros(20,1);
priorofeachclass=zeros(20,1);
traininput=dlmread('train.data');
trainl=dlmread('train.label');
vocabulary=textread('vocabulary.txt','%s');

%size(vocabulary);

% Prior


for i=1:20
    totaldocsperclass(i,1)=sum(trainl(:,1)== i);
end

for i=1:20
    priorofeachclass(i,1)=totaldocsperclass(i,1)/traintotaldocs; % P(Y) of each label
end
 
% Likeihood

wcc=zeros(20,61188);   %Count of wi in cj
for i=1:1467345
  wcc(trainl(traininput(i,1)),traininput(i,2))=wcc(trainl(traininput(i,1)),traininput(i,2))+traininput(i,3);
end

totwincls=sum(wcc,2); %Total words in cj
alpha=1+beta;
p_xi_y=zeros(20,61188);  %P(Xi/Yj) 
for i=1:20
    for j=1:61188
        p_xi_y(i,j)=(wcc(i,j)+(alpha-1))/(totwincls(i)+((alpha-1)*vocablist)); %MAP
    end
end

%Testing classifier with the test data.

testinput=dlmread('test.data');
testl=dlmread('test.label');
wcc_test=zeros(7505,61188);

for i=1:967874   %# of Xi new.
  wcc_test(testinput(i,1),testinput(i,2))= wcc_test(testinput(i,1),testinput(i,2))+testinput(i,3);
end

p_xi_y_transpose=(log2(p_xi_y))';
yp_xi_y=wcc_test*p_xi_y_transpose;
log_priorofeachclass=log2(priorofeachclass);
transpose_yp_xi_y=(yp_xi_y)';   %log2(P(Xi/Yj)

final_new_prior=zeros(20,7505);
for i=1:7505
final_new_prior(:,i)=log_priorofeachclass(:,1);   %log2P(Yj)
end


y_testdata=final_new_prior+transpose_yp_xi_y;    %log2P(Yj)+Sum((#of Xi new)log2(P(Xi/Yj))
 [argvalue, argmax] = max(y_testdata);
argmax=(argmax)';    %Ynew=argmax(log2P(Yj)+Sum((#of Xi new)log2(P(Xi/Yj)))

predict_true = 0;  %Caluclating accuracy.
for i=1:7505
    if(testl(i,1)==argmax(i,1))
        predict_true=predict_true+1;
    end
end

conf_mtx=zeros(20,20);

for i=1:7505
  conf_mtx(testl(i,1),argmax(i,1))= conf_mtx(testl(i,1),argmax(i,1))+1;
end

conf_mtx

%Ranking the vocablist based on proposed method and and printing top 100
%words.
if(input==1) %Executes only if input variable is set to 1 from main.
    Exy=zeros(20,vocablist);
    Px=zeros(1,vocablist);
    for i=1:vocablist   
        Exy(:,i)=(-p_xi_y(:,i).*log2(p_xi_y(:,i))-(1-p_xi_y(:,i)).*log2(1-p_xi_y(:,i))).*priorofeachclass; %Exy=-p(x/y)log2(p(x/y))-(1-p(x/y))*log2(1-p(x/y))*p(y)
        Px(i)=sum(p_xi_y(:,i).*priorofeachclass); %Px=Sum(p(x/y)*p(y))  
    end

    Ex=-Px.*log2(Px)-(1-Px).*log2(1-Px);
    Ixy=(Ex-sum(Exy))';  %Ixy=Ex-Exy
    [temp,index1]=sort(Ixy,'descend');  %Extracting the index values of words in high ranking order
    for i=1:100
        top100list(i,1)=vocabulary(index1(i,1),1);  %Printing top 100 words based on the ranks.
    end
    top100list
end
acc=(predict_true/7505)*100;  %function prj2 returns accuracy value.
end
