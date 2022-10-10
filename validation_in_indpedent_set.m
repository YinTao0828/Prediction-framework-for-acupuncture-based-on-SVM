%% This code is applied to test the rleability of the identified features in predicting acupuncture efficacy
%% SVC
clear all; clc;
NumROI = 35; % nodes of your functional brain network
path = 'your_pathway\matrix'; % pathway of the independent validation set
file = dir([path,filesep, '*.mat']);

% load the matrix of the patients 
conn_msk = ones(NumROI); 
Ind_01 = find(triu(ones(NumROI),1));
Ind_02 = find(conn_msk(Ind_01) ~= 0);
rsFC_data = zeros(length(file), length(Ind_02));
for i = 1:length(file)
    load([path,filesep, file(i).name])
    rsFC_data(i,:) = R(Ind_01(Ind_02)); 
end

% load the clincial data of the patients
load('your_pathway\clinical_feature.txt'); % clinical feature of the independent validation set
data_all = [rsFC_data,clinical_feature];

% load the feature mask
load feature_mask.mat;
Ind = find(cons_feature_mask ~= 0);
data_all = data_all(:,Ind);
clear clinical_feature conn_msk rsFC_data file i path R theROITimeCourses cons_feature_mask;

% load the label
load('your_pathway\label.txt');

% establish SVC models 
permut = 100;
h = waitbar(0,'please wait..'); 
for mn = 1:permut
    waitbar(mn/permut,h,['repetition:',num2str(mn),'/',num2str(permut)]);
    k =10;
    indices = crossvalind('Kfold',size(data_all,1),k);  
    for  i = 1:k
         test = (indices == i); train = ~test;
         train_data = data_all(train,:);
         train_label = label(train,:);
         test_data = data_all(test,:);
         test_label = label(test,:); 
         cmd = ['-s 0, -t 0, -c 1'];
         model = svmtrain(train_label,train_data, cmd);% SVC linear kernal
         [predicted_label, accuracy, deci] = svmpredict(test_label,test_data,model);
         acc(i,1) = accuracy(1);
         deci_value(test,1) = deci;
         predicted(test,1) = predicted_label; % the predicted label of each patient
         clear  test_data  train_data test_label train_label model;
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
end 

% Permutation test for ACC and AUC
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
         cmd = ['-s 0, -t 0, -c 1'];
         model = svmtrain(train_label,train_data, cmd);% SVC linear kernal
         [predicted_label, accuracy, deci] = svmpredict(test_label,test_data,model);
         acc_r(j,1) = accuracy(1);
         deci_value_r(test,1) = deci;
         clear  test_data  train_data test_label train_label model;
    end
    acc_rand(i) = mean(acc_r);
    [~,~,~,AUC_r] = perfcurve(label,deci_value_r,1);
    AUC_rand(i) = AUC_r;
    clear randlabel acc_r label_r AUC deci_value;
end
close(h)
% Compare the randomized ACC and AUC obtained by the permutation test with the minimum ACC and AUC obtained by the model in 100 iterations to determine the significance of the predicted results
p_ACC = mean(abs(acc_rand) >= min(ACC_permut)); 
p_AUC = mean(abs(AUC_rand) >= min(AUC_permut));

%% SVR
clear all; clc;
NumROI = 35; % nodes of your functional brain network
path = 'your_pathway\matrix';
file = dir([path,filesep, '*.mat']);

% load the matrix of the patients 
conn_msk = ones(NumROI); 
Ind_01 = find(triu(ones(NumROI),1));
Ind_02 = find(conn_msk(Ind_01) ~= 0);
rsFC_data = zeros(length(file), length(Ind_02));
for i = 1:length(file)
    load([path,filesep, file(i).name])
    rsFC_data(i,:) = R(Ind_01(Ind_02)); 
end

% load the clincial data of the patients
load('your_pathway\clinical_feature.txt');
data_all = [rsFC_data,clinical_feature];

% load the feature mask
load feature_mask.mat;
Ind = find(cons_feature_mask ~= 0);
data_all = data_all(:,Ind);
clear clinical_feature conn_msk rsFC_data file i path R theROITimeCourses cons_feature_mask;

% load the label
load('your_pathway\label.txt');

% establish SVR models 
permut = 100;
h = waitbar(0,'please wait..'); 
for mn = 1:permut
    waitbar(mn/permut,h,['repetition:',num2str(mn),'/',num2str(permut)]);
    predictive_value = zeros(size(data_all,1),1);
    k =10;
    indices = crossvalind('Kfold',size(data_all,1),k);  
    for  i = 1:k
         test = (indices == i); train = ~test;
         train_data = data_all(train,:);
         train_label = label(train,:);
         test_data = data_all(test,:);
         test_label = label(test,:); 
         cmd = ['-s 3, -t 0, -c 1'];
         model = svmtrain(train_label,train_data, cmd);% SVR linear kernal
         prediction = svmpredict(test_label,test_data,model);
         predictive_value(indices == i) = prediction;
         clear  test_data  train_data test_label train_label model
    end
    
    R = corr(predictive_value, label); % calculate R
    R2 = R*R;
    mse = sum((predictive_value - label).^2)/length(label);  % calculate mean square error


    R2_permut(mn,1) = R*R;
    MSE_permut(mn,1) = mse;
    predictive_value_permut(mn,:) = predictive_value;
end

 % Permutation test for R2 and mse 
    Nsloop = 1000;
    R_rand = zeros(Nsloop,1);
    mse_rand = zeros(Nsloop,1);
    for i = 1:Nsloop
        randlabel = randperm(size(data_all,1));
        label_r  = label(randlabel);
        predictive_value_r = zeros(size(data_all,1),1);
        k =10;
        indices2 = crossvalind('Kfold',size(data_all,1),k);
        for m = 1:k
            test = (indices2 == m); train = ~test;
            train_data = data_all(train,:);
            train_label = label_r(train,:);
            test_data = data_all(test,:);
            test_label = label_r(test,:);
            model = svmtrain(train_label,train_data, '-s 3, -t 0, -c 1');% SVR linear kernal
            predicted_r = svmpredict(test_label,test_data,model); 
            predictive_value_r(indices2 == m) = predicted_r;
            clear  test_data  train_data test_label train_label model predicted_r
         end  
            R_rand(i,1) = corr(predictive_value_r, label_r);
            mse_rand(i,1) = sum((predictive_value_r - label_r).^2)/length(label_r);
            clear randlabel label_r predictive_value_r
    end
close(h)
% Compare the randomized R2 and MSE obtained by the permutation test with the minimum R2 and MSE  obtained by the model in 100 iterations to determine the significance of the predicted results
p_R = mean(abs(R_rand) >= min(R2_permut)); 
p_MSE = mean(abs(mse_rand) >= min(MSE_permut));

