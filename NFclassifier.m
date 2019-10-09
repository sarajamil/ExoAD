function class1 = NFclassifier(data)

load models1NF
load mRMRfeatures1NF

Xraw{1,1} = data{1,1}(1:100,:);

XFeaturesData1 = featureSelectionWin2(Xraw);

n1 = size(mRMRfeatures1,2);

XFeatures_mrmr1 = zeros(1,n1);
for j = 1:n1 %mRMR nFeatures
    XFeatures_mrmr1(1,j) = XFeaturesData1(1,mRMRfeatures1(1,j));
end

class1 = svmclassify(models1NF, XFeatures_mrmr1);

end