% SVM Two Classifiers Train Test
clear all; close all; clc;
% NOTE:
% CLENCH = GRASP
% NF = Extension/Flexion classifier
% CP = Grasp/Pinch classifier

load XrawData
load YData

[YDataNF, YDataCP, idxCP] = TwoClassifiers3(YData);
XrawDataCP = XrawData(idxCP);

%% Features of all data for cvpartition
XFeaturesData1 = featureSelectionWin2(XrawData); %2350 x 204
XFeaturesData2 = featureSelectionWin2(XrawDataCP); %1166 x 204

%% Partition into training and testing
% Get the index for testing/training cross validation
CVO1 = cvpartition(YDataNF,'Kfold',10);
err1 = zeros(CVO1.NumTestSets,1);
CVO2 = cvpartition(YDataCP,'Kfold',10);
err2 = zeros(CVO2.NumTestSets,1);

%% NF Classifier 
display('MIQ');
display('NF Classifier:');
% for n1 = 5:20
%     display(' ');
%     display(['Number of mRMR features: ', num2str(n1)]);
for i = 1:CVO1.NumTestSets
    trainIdx1 = CVO1.training(i);
    testIdx1 = CVO1.test(i);
    % NF
    Xtrain1 = XFeaturesData1(trainIdx1,:);
    Xtest1 = XFeaturesData1(testIdx1,:);
    Ytrain1 = YDataNF(trainIdx1,:);
    Ytest1 = YDataNF(testIdx1,:);
    
    nFeatures1 = 15;
       
    mRMRfeatures1 = mrmr_miq_d(Xtrain1, Ytrain1, nFeatures1);
%     save mRMRfeatures1NF mRMRfeatures1
    
    Xfeaturestrain_mrmr1 = zeros(size(Xtrain1,1),nFeatures1);
    Xfeaturestest_mrmr1 = zeros(size(Xtest1,1),nFeatures1);
        
    for j = 1:nFeatures1
        Xfeaturestrain_mrmr1(:,j) = Xtrain1(:,mRMRfeatures1(j));
        Xfeaturestest_mrmr1(:,j) = Xtest1(:,mRMRfeatures1(j));
    end
    
    %% SVM Train
    tic;
    models1NF = svmtrain(Xfeaturestrain_mrmr1, Ytrain1,'kernel_function','rbf');
%     save models1NF models1NF
    toc;
    
    test1 = svmclassify(models1NF,Xfeaturestest_mrmr1);
    err1(i) = sum(Ytest1 ~= test1); %mis-classification rate
    display(['The error rate 1 is ', num2str(100*(err1(i)/length(Ytest1))), '%']);
       
    % 'kernel_function' = 'rbf'
    % 'rbf_sigma' - scaling factor in the gaussian radial basis function
    % default = 1
    % 'method' - method used to find the hyperplane
    % default = 'SMO' - Sequential minimal optimization (L1 soft margin SVM)
    %           'LS' - least squares (L2 soft-margin SVM)
    % 'autoscale'
    
end

cverr1 = 100*(sum(err1)/sum(CVO1.TestSize));
display(['>>> The total error rate 1 is ', num2str(cverr1), '%']);

% end

%% CP Classifier
clear i j;

display('--------------------------------------------');
display('MIQ');
display('CP Classifier:');
% for n2 = 5:20
%     display(' ');
%     display(['Number of mRMR features: ', num2str(n2)]);
for i = 1:CVO2.NumTestSets
    trainIdx2 = CVO2.training(i);
    testIdx2 = CVO2.test(i);
    % CP
    Xtrain2 = XFeaturesData2(trainIdx2,:);
    Xtest2 = XFeaturesData2(testIdx2,:);
    Ytrain2 = YDataCP(trainIdx2,:);
    Ytest2 = YDataCP(testIdx2,:);
    
    nFeatures2 = 10;
    
    mRMRfeatures2 = mrmr_miq_d(Xtrain2, Ytrain2, nFeatures2);
%     save mRMRfeatures2CP mRMRfeatures2
    
    Xfeaturestrain_mrmr2 = zeros(size(Xtrain2,1),nFeatures2);
    Xfeaturestest_mrmr2 = zeros(size(Xtest2,1),nFeatures2);
    
    for j = 1:nFeatures2
        Xfeaturestrain_mrmr2(:,j) = Xtrain2(:,mRMRfeatures2(j));
        Xfeaturestest_mrmr2(:,j) = Xtest2(:,mRMRfeatures2(j));
    end
    
    %% SVM Train
    tic;
    models2CP = svmtrain(Xfeaturestrain_mrmr2, Ytrain2,'kernel_function','rbf');
%     save models2CP models2CP
    toc;
    
    test2 = svmclassify(models2CP,Xfeaturestest_mrmr2);
    err2(i) = sum(Ytest2 ~= test2); %mis-classification rate
    display(['The error rate 2 is ', num2str(100*(err2(i)/length(Ytest2))), '%']);
    
end

cverr2 = 100*(sum(err2)/sum(CVO2.TestSize));
display(['>>> The total error rate 2 is ', num2str(cverr2), '%']);

% end
