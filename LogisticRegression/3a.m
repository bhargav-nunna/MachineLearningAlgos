

% Reading the input data
baseDir='/Users/.../..../sound/genres/';
myFolder = {'classical';'jazz';'CrossValCountry';'pop';'rock';'metal'};

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

TotConfusionMatrix=zeros(6,6);
X=NormalizedValues';


j=1;
for i=1:600
    correc(1,i)=j;
    if(rem(i,100)==0)
        j=j+1;
    end
end

W=zeros(6,1001);
%10 fold corss validation data


for CrossValCount=1:10
    CrossValInd= crossvalind('Kfold', 600, 10);
    W=zeros(6,1001);
    eta=0.01;
    
    k=1;
    l=1;
    %Seperating of Testing and Training Data
    for i=1:600
        if(CrossValInd(i,1)==CrossValCount)
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
    
    for k=1:1001
        
            TrainData(:,k)=TrainData(:,k)/max(TrainData(:,k));
            TestData(:,k)=TestData(:,k)/max(TestData(:,k));
    
    end
   
    

%Training of the classifier
    for i=1:200
    
        Probs=exp(W*TrainData');
        for k=1:540
            Probs(:,k)=Probs(:,k)/max(Probs(:,k));
        end
        [maximum,index]=max(Probs);
    
        W=W+eta*(((ActClassesTrain-Probs)*TrainData)-(Penalty*W));
        eta=0.01/(1+i/200);
    end

    %Testing the classifier
    TestProbs=exp(W*TestData');
    
        for k=1:60
            TestProbs(:,k)=TestProbs(:,k)/max(TestProbs(:,k));
        end
        [Testmaximum,Testindex]=max(TestProbs);


    TestingAccuracy=0;
    for j=1:60
        if(TestActLabel(j)==Testindex(j))
            TestingAccuracy=TestingAccuracy+1;
        end
    end
    confusionMatrix=zeros(6,6);
    %Caluclating Confusion Matrix
    for i=1:60
        
        confusionMatrix(TestActLabel(1,i),Testindex(1,i))=confusionMatrix(TestActLabel(1,i),Testindex(1,i))+1;
        
    end
    TestingAccuracyr(CrossValCount)=(TestingAccuracy/60)*100;
    TotConfusionMatrix=TotConfusionMatrix+confusionMatrix; %Total confusionMatrix for 10 fold
end