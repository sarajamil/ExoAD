function [Xfeatures, Xenv] = featureSelectionWin2(data)
% Filter
Fs = 200; % Sample rate of the EMG data

% First butterworth fc = 2, n = 3
N1 = 3;
Fc1 = 2;
[B1,A1] = butter(N1,(Fc1/(Fs/2)), 'low');

% Second butterworth fc = 5, n = 10
N2 = 10;
Fc2 = 5;
[B2,A2] = butter(N2,(Fc2/(Fs/2)), 'low');

trials = length(data); 
channels = 8;

% Enveloping
for i = 1:trials
    rectify_clench{i,1} = abs(data{i,1});
    rectify_clench{i,2} = abs(data{i,1});
%     rectify_clench{i,3} = abs(data{i,1});
    envelope_clench{i,1} = filter(B1,A1,rectify_clench{i,1});
    envelope_clench{i,2} = filter(B2,A2,rectify_clench{i,2});
%     envelope_clench{i,3} = filter(B3,A3,rectify_clench{i,3});
    Xenv{i,1} = envelope_clench{i,1};
    Xenv{i,2} = envelope_clench{i,2};
%     Xenv{i,3} = envelope_clench{i,3};
end

% Features:
% variance, standard deviation, mean absolute value
% taken over 100 sample data for ALL CHANNELS
% and taken over each channel
Xfeatures = [];

for i = 1:trials
    var_channels = zeros(1,16);
    std_channels = zeros(1,16);
    mav_channels = zeros(1,16);
    for k = 1:2
        for j = 1:channels
            var_channels((k-1)*channels+j) = var(Xenv{i,k}(:,j));
            std_channels((k-1)*channels+j) = std(Xenv{i,k}(:,j));
            mav_channels((k-1)*channels+j) = mean(Xenv{i,k}(:,j));
        end
    end
    
    % slope for third filter only? 1:25? on data?
    var_slope = var(Xenv{i,2}(76:100,:)) - var(Xenv{i,2}(51:75,:));
    std_slope = std(Xenv{i,2}(76:100,:)) - std(Xenv{i,2}(51:75,:));
    mav_slope = mean(Xenv{i,2}(76:100,:)) - mean(Xenv{i,2}(51:75,:));
    max_slope = max(Xenv{i,2}(76:100,:)) - min(Xenv{i,2}(76:100,:));

    cc_raw = corrcoef(data{i,1});
    cc_env1 = corrcoef(Xenv{i,1});    
    cc_env2 = corrcoef(Xenv{i,2}); 
%     cc_env3 = corrcoef(Xenv{i,3}); 
    corrcoef_raw = (cc_raw(logical(triu(true(size(cc_raw)))-eye(8))))';
    corrcoef_env1 = (cc_env1(logical(triu(true(size(cc_env1)))-eye(8))))';
    corrcoef_env2 = (cc_env2(logical(triu(true(size(cc_env2)))-eye(8))))';
%     corrcoef_env3 = (cc_env3(logical(triu(true(size(cc_env3)))-eye(8))))';
    
    feature_fft = zeros(1,40);
    for j = 1:channels
        Xfeaturefft{i,1}(:,j) = fftshift(abs(fft(data{i,1}(:,j), 200)));
    end
    feature_fft(1:8) = mean(abs(Xfeaturefft{i,1}(1:20,:)));
    feature_fft(9:16) = mean(abs(Xfeaturefft{i,1}(21:40,:)));
    feature_fft(17:24) = mean(abs(Xfeaturefft{i,1}(41:60,:)));
    feature_fft(25:32) = mean(abs(Xfeaturefft{i,1}(61:80,:)));
    feature_fft(33:40) = mean(abs(Xfeaturefft{i,1}(81:100,:)));
    
%     xcorr_raw = xcorr(data{i,1});
%     xcorr_env1 = xcorr(Xenv{i,1});
%     xcorr_env2 = xcorr(Xenv{i,2});
%     xcorr_feature = [...
%         xcorr_raw(100)...
%         xcorr_raw(120,:) xcorr_raw(140,:)...
%         xcorr_env1(100,:) xcorr_env1(120,:) xcorr_env1(140,:)...
%         xcorr_env2(100,:)...
%         xcorr_env2(120,:) xcorr_env2(140,:)...
%         ];
%     
%     xcorr_fft_raw = zeros(1,320);
%     xcorr_fft_env1 = zeros(1,320);
%     xcorr_fft_env2 = zeros(1,320);
%     for j = 1:(channels*channels)
%         Xfftxcorr{i,1}(:,j) = fftshift(abs(fft(xcorr_raw(:,j))));
%         Xfftxcorr{i,2}(:,j) = fftshift(abs(fft(xcorr_env1(:,j))));
%         Xfftxcorr{i,3}(:,j) = fftshift(abs(fft(xcorr_env2(:,j))));
%     end
%     xcorr_fft_raw(1:64) = mean(abs(Xfftxcorr{i,1}(1:20,:)));
%     xcorr_fft_raw(65:128) = mean(abs(Xfftxcorr{i,1}(21:40,:)));
%     xcorr_fft_raw(129:192) = mean(abs(Xfftxcorr{i,1}(41:60,:)));
%     xcorr_fft_raw(193:256) = mean(abs(Xfftxcorr{i,1}(61:80,:)));
%     xcorr_fft_raw(257:320) = mean(abs(Xfftxcorr{i,1}(81:100,:)));
%     xcorr_fft_env1(1:64) = mean(abs(Xfftxcorr{i,1}(1:20,:)));
%     xcorr_fft_env1(65:128) = mean(abs(Xfftxcorr{i,1}(21:40,:)));
%     xcorr_fft_env1(129:192) = mean(abs(Xfftxcorr{i,1}(41:60,:)));
%     xcorr_fft_env1(193:256) = mean(abs(Xfftxcorr{i,1}(61:80,:)));
%     xcorr_fft_env1(257:320) = mean(abs(Xfftxcorr{i,1}(81:100,:)));
%     xcorr_fft_env2(1:64) = mean(abs(Xfftxcorr{i,1}(1:20,:)));
%     xcorr_fft_env2(65:128) = mean(abs(Xfftxcorr{i,1}(21:40,:)));
%     xcorr_fft_env2(129:192) = mean(abs(Xfftxcorr{i,1}(41:60,:)));
%     xcorr_fft_env2(193:256) = mean(abs(Xfftxcorr{i,1}(61:80,:)));
%     xcorr_fft_env2(257:320) = mean(abs(Xfftxcorr{i,1}(81:100,:)));
%     
    Xfeatures = [Xfeatures;...
        var_channels std_channels mav_channels...
        var_slope std_slope mav_slope max_slope...
        corrcoef_raw corrcoef_env1 corrcoef_env2...
        feature_fft...
%         xcorr_feature...
%         xcorr_fft_raw...
        ];
end

end