
% Reading the input data
baseDir='/Users/.../..../sound/genres/';

myFolder = {'classical';'jazz';'CrossValCountry';'pop';'rock';'metal'};
          
 ActualClass=zeros(6,600);
ActualClass(1,1:100)=ones(1,100);
ActualClass(2,101:200)=ones(1,100);
ActualClass(3,201:300)=ones(1,100);
ActualClass(4,301:400)=ones(1,100);
ActualClass(5,401:500)=ones(1,100);
ActualClass(6,501:0600)=ones(1,100);
eta=0.01;
Penalty=0.001;      

for fIndex=1:6
    fileName=strcat(baseDir,myFolder(fIndex));
    filePattern = fullfile(char(fileName), '*.wav');
    wavFiles = dir(filePattern);
    for k = 1:length(wavFiles)
        baseFileName = wavFiles(k).name;
        fullFileName = fullfile(fileName, baseFileName);
       
         %[speech,fs]=audioread(a(k).name);
        %[ MFCCs, FBEs, frames ] = mfcc( speech, fs, Tw, Ts, alpha, @hamming, R, M, C, L );
        Mff=mfccReading(char(fullFileName));
        [row,col]=size(Mff);
        for i=1:13
            M(i,:)=mean(Mff(i,ceil(col*(1/10)):ceil(col*(9/10))));
        end
    
        inputsongs(:,k+((fIndex-1)*100))=M(:,1);
    end
end

NormalizedValues=ones(14,600); %for bias

        

%Normalizing the input data.
%inputsongs=fft(inputsongs);

for k=1:600
    NormalizedValues(2:14,k)=inputsongs(1:13,k); %./max(inputsongs(1:1000,k));
end

X=NormalizedValues';
TotConfusionMat=zeros(6,6);



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
    W=zeros(6,14);
    eta=0.01;
    
    k=1;
    l=1;
    %Seperating of Testing and Training Data
    for i=1:600
        if(CrossValInd(i,1)==CrossValCount)
            TestData(k,:)=X(i,:);
            ActualClassTest(:,k)=ActualClass(:,i);
            TestActLabel(1,k)=correc(1,i);
            k=k+1;
        else
            TrainingData(l,:)=X(i,:);
            ActualClassTrain(:,l)=ActualClass(:,i);
            TrainActLabel(1,l)=correc(1,i);
            l=l+1;
        end
    end
    
    for k=1:14
        
            TrainingData(:,k)=TrainingData(:,k)/max(TrainingData(:,k));
      
            TestData(:,k)=TestData(:,k)/max(TestData(:,k));
    
    end
   


%Training of the classifier
    for i=1:300
    
        Probs=exp(W*TrainingData');
        for k=1:540
            Probs(:,k)=Probs(:,k)/max(Probs(:,k));
        end
        [maximum,index]=max(Probs);
    
        W=W+eta*(((ActualClassTrain-Probs)*TrainingData)-Penalty*W);
        eta=0.01/(1+i/300);
    end

    %Testing the classifier
    TestProbs=exp(W*TestData');
    
        for k=1:60
            TestProbs(:,k)=TestProbs(:,k)/max(TestProbs(:,k));
        end
        [Testmaximum,Testindex]=max(TestProbs);

    Trainacc=0;
    for j=1:540
        if(TrainActLabel(j)==index(j))
            Trainacc=Trainacc+1;
        end
    end

    Testacc=0;
    for j=1:60
        if(TestActLabel(j)==Testindex(j))
            Testacc=Testacc+1;
        end
    end
    confmat=zeros(6,6);
    %Caluclating Confusion Matrix
    for i=1:60
        
        confmat(TestActLabel(1,i),Testindex(1,i))=confmat(TestActLabel(1,i),Testindex(1,i))+1;
        
    end
    TestingAccuracy(CrossValCount)=(Testacc/60)*100;
    TotConfusionMat=TotConfusionMat+confmat; %Total confusionMatrix for 10 fold
end