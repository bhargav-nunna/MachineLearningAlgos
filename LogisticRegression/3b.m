
% Reading the input data
baseDir='/Users/.../..../sound/genres/';

myFolder = {'classical';'jazz';'country';'pop';'rock';'metal'};

TotConfusionMat=zeros(6,6);
ActClasses=zeros(6,600);
ActClasses(1,1:100)=ones(1,100);
ActClasses(2,101:200)=ones(1,100);
ActClasses(3,201:300)=ones(1,100);
ActClasses(4,301:400)=ones(1,100);
ActClasses(5,401:500)=ones(1,100);
ActClasses(6,501:0600)=ones(1,100);
eta=0.01;
Penalty=0.001;


for fIndex=1:6
    fileName=strcat(baseDir,myFolder(fIndex));
    filePattern = fullfile(char(fileName), '*.wav');
    wavFiles = dir(filePattern);
    for k = 1:length(wavFiles)
        baseFileName = wavFiles(k).name;
        fullFileName = fullfile(fileName, baseFileName);
        [y,fs]=audioread(char(fullFileName));
        inputsongs(1:1000,k+((fIndex-1)*100))=y(1:1000,1);
    end
end

NormalizedValues=ones(1001,600); %for bias

%Normalizing the input data.
inputsongs=fft(inputsongs);

for k=1:600
    NormalizedValues(2:1001,k)=inputsongs(1:1000,k); %./max(inputsongs(1:1000,k));
end

NewNormalizedVals=NormalizedValues';
%Extracting the best features
sdev=std(NewNormalizedVals(1:100,1:1001));
[val,sortind]=sort(sdev);
Selected1(1,:)=sortind(1:20);

sdev=std(NewNormalizedVals(101:200,1:1001));
[val,sortind]=sort(sdev);
Selected2(1,:)=sortind(1:20);

sdev=std(NewNormalizedVals(201:300,1:1001));
[val,sortind]=sort(sdev);
Selected3(1,:)=sortind(1:20);

sdev=std(NewNormalizedVals(301:400,1:1001));
[val,sortind]=sort(sdev);
Selected4(1,:)=sortind(1:20);

sdev=std(NewNormalizedVals(401:500,1:1001));
[val,sortind]=sort(sdev);
Selected5(1,:)=sortind(1:20);

sdev=std(NewNormalizedVals(501:600,1:1001));
[val,sortind]=sort(sdev);
Selected6(1,:)=sortind(1:20);

Selected=[Selected1,Selected2,Selected3,Selected4,Selected5,Selected6];
NewSelected=unique(Selected);

for i=1:length(NewSelected)
    X(:,i)=NewNormalizedVals(:,NewSelected(i));
end
 
len=length(X);



j=1;
for i=1:600
    correc(1,i)=j;
    if(rem(i,100)==0)
        j=j+1;
    end
end


W=zeros(6,length(NewSelected));
%10 fold corss validation data


for count=1:100
    CrossValInd= crossvalind('Kfold', 600, 10);
    W=zeros(6,length(NewSelected));
    eta=0.01;
    
    k=1;
    l=1;
    %Seperating of Testing and Training Data
    for i=1:600
        if(CrossValInd(i,1)==count)
            TestData(k,:)=X(i,:);
            ActClassesTest(:,k)=ActClasses(:,i);
            TestActLabel(1,k)=correc(1,i);
            k=k+1;
        else
            TrainData(l,:)=X(i,:);
            ActClassesTrain(:,l)=ActClasses(:,i);
            TrainActLabel(1,l)=correc(1,i);
            l=l+1;
        end
    end
    
    for k=1:length(NewSelected)
        
            TrainData(:,k)=TrainData(:,k)/max(TrainData(:,k));
      
            TestData(:,k)=TestData(:,k)/max(TestData(:,k));
    
    end
   
    


%Training of the classifier
    for i=1:10
    
        Probs=exp(W*TrainData');
        for k=1:540
            Probs(:,k)=Probs(:,k)/max(Probs(:,k));
        end
        [maximum,index]=max(Probs);
    
        W=W+eta*(((ActClassesTrain-Probs)*TrainData)-Penalty*W);
        eta=eta/(1+i/10);
    end

    %Testing the classifier
    TestProbs=exp(W*TestData');
    
        for k=1:60
            TestProbs(:,k)=TestProbs(:,k)/max(TestProbs(:,k));
        end
        [Testmaximum,Testindex]=max(TestProbs);

   

    Testacc=0;
    for j=1:60
        if(TestActLabel(j)==Testindex(j))
            Testacc=Testacc+1;
        end
    end
    ConfusionMat=zeros(6,6);
    
    %Caluclating Confusion Matrix
    for i=1:60
        
        ConfusionMat(TestActLabel(1,i),Testindex(1,i))=ConfusionMat(TestActLabel(1,i),Testindex(1,i))+1;
        
    end
    TerstingAccuracy(count)=(Testacc/60)*100;
    TotConfusionMat=TotConfusionMat+ConfusionMat; %Total confusionMatrix for 10 fold
end