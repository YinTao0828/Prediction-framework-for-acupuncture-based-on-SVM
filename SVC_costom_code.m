%% This code is applied to establish SVC prediction models for the prediciton of acupuncture responders and non-responders 
%% The LIBSVM toolbox(https://www.csie.ntu.edu.tw/~cjlin/Libsvm)is needed!
%% If any question, pls contact Tao Yin. Emial: yintao@cdutcm.edu.cn

%% Part 1. Establish SVC classification models under the feature selection thresholds from 0.005-0.5.
clc; clear all;
for f = 1:100 % differnt feature selection thresholds
    clearvars -except f; 
    NumROI = 35; % nodes of your functional brain network
    path = 'your_pathway\matrix'; % Pathways that store the functional brain network matrix (35*35) of all patients
    file = dir([path,filesep, '*.mat']);
    conn_msk = ones(NumROI); 
    Ind_01 = find(triu(ones(NumROI),1));
    Ind_02 = find(conn_msk(Ind_01) ~= 0);
    
    % load the matrix of the patients
    rsFC_data = zeros(length(file), length(Ind_02));
    for i = 1:length(file)
        load([path,filesep, file(i).name])
        rsFC_data(i,:) = R(Ind_01(Ind_02)); % R is the name of every matrix
    end
    clear conn_msk file i path R;
    
    % load the clincial data of the patients
    load('your_pathway\clinical_feature.txt');
    data_all = [rsFC_data,clinical_feature];

    % load the label
    load('your_pathway\label.txt');
    
    % establish SVC models 
    h = waitbar(0,'please wait..');
    permut = 100; % permution for the k-fold cross-validation 
    for mn = 1:permut
        waitbar(mn/permut,h,['repetition:',num2str(mn),'/',num2str(permut)]);
        k = 10; % 10-fold cross-validation 
        indices = crossvalind('Kfold',size(data_all,1),k);
        for  i = 1:k
             test = (indices == i); train = ~test;
             train_data = data_all(train,:);
             train_label = label(train,:);
             test_data = data_all(test,:);
             test_label = label(test,:);
             % Fiter selection method£ºcorr
             [~,p]  = corr(train_data, train_label, 'type','Spearman');
             sigInd = find(p < 0.005*f);
             train_data = train_data(:,sigInd);
             test_data = test_data(:,sigInd);
             cmd = ['-s 0, -t 0']; % linear kernal SVC 
             model = svmtrain(train_label,train_data, cmd); % train the model
             w = model.SVs'*model.sv_coef; % Weights of the support vector
             wMatrix = zeros(NumROI);
             wMatrix(Ind_01(sigInd)) = w;
             wMatrix = wMatrix+wMatrix';
             cons_feature(:,:,i) = wMatrix;
             [predicted_label, accuracy, deci] = svmpredict(test_label,test_data,model); % applied the trained model for prediction
             acc(i,1) = accuracy(1);
             deci_value(test,1) = deci;
             predicted(test,1) = predicted_label; % the predicted label of each patient
             clear  test_data  train_data test_label train_label model
        end
        ACC = mean(acc);
        Sensitivty = sum((label==1)&(predicted==1))/sum(label==1);
        Specificity = sum((label==-1)&(predicted==-1))/sum(label==-1);
        [X,Y,~,AUC] = perfcurve(label,deci_value,1);
        
        % performance of model across the 100 permutations of k-fold cross-validation
        ACC_permut(mn,:) = ACC;
        Sensitivty_premut(mn,:) = Sensitivty;
        Specificity_premut(mn,:) = Specificity;
        AUC_permut(mn,:) = AUC;
        X_permut(mn,:) = X;
        Y_permut(mn,:) = Y;
        
        % Calculation of consensus features' weights
        % The features that retained as support vectors in all 10-fold CV were defined as the consensus features
        cons_feature_mask = double(sum(cons_feature ~= 0,3) == k); 
        cons_feature_mask = sign(mean(cons_feature,3)).*cons_feature_mask;
        cons_feature_mean = mean(cons_feature,3).*double(cons_feature_mask~=0); 
        cons_feature_permut(:,:,mn) = cons_feature_mean;
        cons_feature_mask_permut(:,:,mn) = cons_feature_mask;

        clearvars -except f p_ACC_filter Ind_01 Ind_02 data_all label permut mn h NumROI Nsloop ACC_permut Sensitivty_premut Specificity_premut AUC_permut X_permut Y_permut cons_feature_permut cons_feature_mask_permut;
    end
        close(h)
        clear h mn Ind_01 Ind_02;
        name = ['FD_response_prediction_', num2str(f,'%03d'),'.mat'];
        save(name); % a total of 100 files which contain theresults of prediction under different feature selection thresholds are saved in the current folder
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Part 2. Find the optimal threshold
clear all; clc;
ACC = zeros(100,1);
AUC = zeros(100,1);
SEN = zeros(100,1);
SPE = zeros(100,1);

% Extract the prediction results at each threshold
for f = 1:100
name = ['FD_response_prediction_', num2str(f,'%03d'),'.mat'];
load (name);
ACC_incresefeature = mean(ACC_permut);
ACC(f,1) =  ACC_incresefeature/100;  

AUC_incresefeature = mean(AUC_permut);
AUC(f,1) =  AUC_incresefeature; 

SEN_incresefeature = mean(Sensitivty_premut);
SEN(f,1) =  SEN_incresefeature; 

SPE_incresefeature = mean(Specificity_premut);
SPE(f,1) =  SPE_incresefeature;   
end

% Plot the changes of model performance with thresholds change and identified the optimal threshold
figure;
plot(ACC,'DisplayName','ACC_SID');hold on;plot(AUC,'DisplayName','AUC_SID');plot(SEN,'DisplayName','SEN_SID');plot(SPE,'DisplayName','SPE_SID');hold off;
xlabel('Threshold for feature screening (x0.005)'); title('Changes in classification performance')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Part 3. Build SVC model based on the optimal threshold to predict acupuncture responders and non-responders and conduction permutation test
clc; clear all;
f = 30; % The optimal threshold in the current study ¦Á=0.15
NumROI = 35; % nodes of your functional brain network
path = 'your_pathway\matrix'; % Pathways that store the functional brain network matrix (35*35) of all patients
file = dir([path,filesep, '*.mat']);
conn_msk = ones(NumROI); 
Ind_01 = find(triu(ones(NumROI),1));
Ind_02 = find(conn_msk(Ind_01) ~= 0);

% load the matrix of the patients
rsFC_data = zeros(length(file), length(Ind_02));
for i = 1:length(file)
    load([path,filesep, file(i).name])
    rsFC_data(i,:) = R(Ind_01(Ind_02)); % R is the name of every matrix
end
clear conn_msk file i path R;
    
% load the clincial data of the patients
load('your_pathway\clinical_feature.txt');
data_all = [rsFC_data,clinical_feature];

% load the lable
load('your_pathway\lable.txt');

% establish SVC models 
h = waitbar(0,'please wait..');
permut = 100; % permution for the k-fold cross-validation 
for mn = 1:permut
    waitbar(mn/permut,h,['repetition:',num2str(mn),'/',num2str(permut)]);
    k = 10; % 10-fold cross-validation 
    indices = crossvalind('Kfold',size(data_all,1),k);
    for  i = 1:k
         test = (indices == i); train = ~test;
         train_data = data_all(train,:);
         train_label = label(train,:);
         test_data = data_all(test,:);
         test_label = label(test,:);
         % Fiter selection method£ºcorr
         [~,p]  = corr(train_data, train_label, 'type','Spearman');
         sigInd = find(p < 0.005*f);
         train_data = train_data(:,sigInd);
         test_data = test_data(:,sigInd);
         cmd = ['-s 0, -t 0']; % linear kernal SVC 
         model = svmtrain(train_label,train_data, cmd); % train the model
         w = model.SVs'*model.sv_coef; % Weights of the support vector
         wMatrix = zeros(NumROI);
         wMatrix(Ind_01(sigInd)) = w;
         wMatrix = wMatrix+wMatrix';
         cons_feature(:,:,i) = wMatrix;
         [predicted_label, accuracy, deci] = svmpredict(test_label,test_data,model); % applied the trained model for prediction
         acc(i,1) = accuracy(1);
         deci_value(test,1) = deci;
         predicted(test,1) = predicted_label; % the predicted label of each patient
         clear  test_data  train_data test_label train_label model
    end
    ACC = mean(acc);
    Sensitivty = sum((label==1)&(predicted==1))/sum(label==1);
    Specificity = sum((label==-1)&(predicted==-1))/sum(label==-1);
    [X,Y,~,AUC] = perfcurve(label,deci_value,1);
    
    % performance of model across the 100 permutations of k-fold cross-validation
    ACC_permut(mn,:) = ACC;
    Sensitivty_premut(mn,:) = Sensitivty;
    Specificity_premut(mn,:) = Specificity;
    AUC_permut(mn,:) = AUC;
    X_permut(mn,:) = X;
    Y_permut(mn,:) = Y;
    
    % Calculation of consensus features' weights
    % The features that retained as support vectors in all 10-fold CV were defined as the consensus features
    cons_feature_mask = double(sum(cons_feature ~= 0,3) == k); 
    cons_feature_mask = sign(mean(cons_feature,3)).*cons_feature_mask;
    cons_feature_mean = mean(cons_feature,3).*double(cons_feature_mask~=0); 
    cons_feature_permut(:,:,mn) = cons_feature_mean;
    cons_feature_mask_permut(:,:,mn) = cons_feature_mask;

    clearvars -except f p_ACC_filter Ind_01 Ind_02 data_all label permut mn h NumROI Nsloop ACC_permut Sensitivty_premut Specificity_premut AUC_permut X_permut Y_permut cons_feature_permut cons_feature_mask_permut;
end
    close(h)
    clear h mn Ind_01 Ind_02;
%% Permutation test for ACC and AUC
Nsloop = 1000;
acc_rand = zeros(Nsloop,1);
AUC_rand = zeros(Nsloop,1);
h = waitbar(0,'please wait..');
for i=1:Nsloop
    waitbar(i/Nsloop,h,['repetition:',num2str(i),'/',num2str(Nsloop)]);
    randlabel = randperm(length(label));
    label_r  = label(randlabel);
    k =10;
    indices = crossvalind('Kfold',size(data_all,1),k);  
    for j = 1:k
         test = (indices == j); train = ~test;
         train_data = data_all(train,:);
         train_label = label_r(train,:);
         test_data = data_all(test,:);
         test_label = label_r(test,:);
         [~,p]  = corr(train_data, train_label, 'type','Spearman');
         sigInd = find(p < 0.005*f);
         train_data = train_data(:,sigInd);
         test_data = test_data(:,sigInd); 
         model = svmtrain(train_label,train_data, '-s 0, -t 0');% SVC linear kernal
         [predicted_label, accuracy, deci] = svmpredict(test_label,test_data,model);
         acc_r(j,1) = accuracy(1);
         deci_value(test,1) = deci;
         clear  test_data  train_data test_label train_label model;
    end
    acc_rand(i) = mean(acc_r);
    [~,~,~,AUC] = perfcurve(label,deci_value,1);
    AUC_rand(i) = AUC;
    clear randlabel acc_r label_r AUC deci_value;
end
close(h)
% Compare the randomized ACC and AUC obtained by the permutation test with the minimum ACC and AUC obtained by the model in 100 iterations to determine the significance of the predicted results
p_ACC = mean(abs(acc_rand) >= min(ACC_permut)); 
p_AUC = mean(abs(AUC_rand) >= min(AUC_permut));

%% Identify predicting features and their weights
% consensus features that occurred in all 100 iterations were identified as the critical predicting features 
cons_feature_mask = double(sum(cons_feature_mask_permut ~= 0,3)  == permut); 
cons_feature = mean(cons_feature_permut,3).*cons_feature_mask;
imagesc(cons_feature)
hold on; plot([6.5,6.5],[0.5,35.5],'w','linewidth',2);
hold on; plot([16.5,16.5],[0.5,35.5],'w','linewidth',2);
hold on; plot([26.5,26.5],[0.5,35.5],'w','linewidth',2);
hold on; plot([0.5,35.5],[6.5,6.5],'w','linewidth',2);
hold on; plot([0.5,35.5],[16.5,16.5],'w','linewidth',2);
hold on; plot([0.5,35.5],[26.5,26.5],'w','linewidth',2);
axis off; 
