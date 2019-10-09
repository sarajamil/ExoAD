function class2 = CPclassifier(data)

load models2CP
load mRMRfeatures2CP

Xraw{1,1} = data{1,1}(1:100,:);

XFeaturesData2 = featureSelectionWin2(Xraw);

n2 = size(mRMRfeatures2,2);

XFeatures_mrmr2 = zeros(1,n2);
for j = 1:n2 %mRMR nFeatures
    XFeatures_mrmr2(1,j) = XFeaturesData2(1,mRMRfeatures2(1,j));
end

class2 = svmclassify(models2CP, XFeatures_mrmr2);

end