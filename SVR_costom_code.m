%% This code is applied to establish SVR prediction models for the improvement of clinical symptom of patients
%% The predicting features identified in SVC prediction is used in the prediction analysis
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

%% Permutation test for R2 and mse 
Nsloop = 1000;
R_Nsloop = zeros(Nsloop,1);
mse_Nsloop = zeros(Nsloop,1);
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
        R_Nsloop(i,1) = corr(predictive_value_r, label_r);
        mse_Nsloop(i,1) = sum((predictive_value_r - label_r).^2)/length(label_r);
        clear randlabel label_r predictive_value_r
end
close(h)
% Compare the randomized R2 and MSE obtained by the permutation test with the minimum R2 and MSE  obtained by the model in 100 iterations to determine the significance of the predicted results
p_R = mean(abs(R_Nsloop) >= min(R2_permut)); 
p_MSE = mean(abs(mse_Nsloop) >= min(MSE_permut));
