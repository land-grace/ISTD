
function [out,psi1,psi2,psi3]=Copy_of_my_INNE_IRST(imgo)% 自适应AP用 t=40 psi=16
img_gray=double(imgo);
img_gray(:,1)=img_gray(:,2);

%% 局部灰度强度
img_pad = padarray(img_gray,[5 5],'replicate','both');  %---------扩展图像---------
[Isum,Imax]=mI(img_pad);
Isum=Isum(6:end-5,6:end-5);
Imax=Imax(6:end-5,6:end-5);

%% 局部梯度强度
GgropM=LocalGrad(img_gray);
GgropMN=NormalizeData(GgropM);

%% 原始对称能量
% 参数设置
nscale = 4;     % 4个尺度
norient = 6;    % 6个方向
minWaveLength = 2;  % 最小波长 >=2
mult = 1.5;           % 尺度倍增因子
sigmaOnf = 0.45;    % 滤波器带宽
% 计算相位一致性图
[~, ~, TE, ~] = phasesym(img_pad, nscale, norient, minWaveLength, mult, sigmaOnf,'polarity',-1);
TE=TE(6:end-5,6:end-5);

%% 自适应PSI
maxTE=max(TE(:));conTE=maxTE-TE;
num_Isum = estimatePeaks(Imax)+1;
num_TE = estimatePeaks(conTE)+1;
% figure; imshow(GgropM(:,:,4), []);title('GgropM(:,:,4)');
num_G=0;nubGlist=zeros(1,24);
for i=1:24
    nubGlist(i)=estimatePeaks(GgropM(:,:,i));
end
num_G = max(nubGlist)+1; 

%%
% 获取图像的长宽
    [m,n]=size(img_gray); pn=m*n;
    Cpg2 = reshape(Isum(:, :), m*n, 1);  
    Cpg6 = reshape(Imax(:, :), m*n, 1);  
    Cpg7 = reshape(GgropMN(:,:,:), m*n, 24);   
    Cpg8 = reshape(TE(:,:,:), m*n, 1);    
%-----距离矩阵----
    zby=zeros(m,n);zbx=zeros(m,n);
    for i=1:m
        zby(i,:)=i;
    end
    for i=1:n
        zbx(:,i)=i;
    end
    Czby = reshape(zby, m*n, 1);
    Czbx = reshape(zbx, m*n, 1);
Czby = (Czby-min(Czby))./(max(Czby)-min(Czby));%
Czbx = (Czbx-min(Czbx))./(max(Czbx)-min(Czbx));%
Cpg2 = (Cpg2-min(Cpg2))./(max(Cpg2)-min(Cpg2));%
Cpg6 = (Cpg6-min(Cpg6))./(max(Cpg6)-min(Cpg6));%多尺度差分归一化
Cpg7 = (Cpg7-min(Cpg7))./(max(Cpg7)-min(Cpg7));%梯度归一化
Cpg8 = (Cpg8-min(Cpg8))./(max(Cpg8)-min(Cpg8));%能量归一化
%% iNNE
t = 180; % max ensemble size
psi1 = max(round(num_Isum^0.5),8); psi2=max(round(num_G^0.5),16); psi3=max(round(num_TE^0.5),8);

data1=[Cpg2 Cpg6 1.5*Czby 1.5*Czbx];
[Score,~]= iNNE(data1,data1,t,psi1,m,n);
mScore=Score;
iNNE_results1=reshape(mScore,m,n);


data3=[Cpg7 8*Czby 8*Czbx]; 
[Score,~]= iNNE(data3,data3,t,psi2,m,n);
mScore3=Score;
iNNE_results3=reshape(mScore3,m,n);


data4=[Cpg8 0.7*Czby 0.7*Czbx]; %对称能量
[Score,~]= iNNE(data4,data4,t,psi3,m,n);
mScore4=Score;
iNNE_results4=reshape(mScore4,m,n);

%% 闭操作+AP 
fusion=iNNE_results1.*iNNE_results3.*iNNE_results4;
final=closeAP(fusion);%
out=final./max(final(:));
lamuda=0.6;
Thr=lamuda*max(out(:))+(1-lamuda)*mean(out(:));
out(out<Thr)=0;


%%
function [ data ] = NormalizeData( data )
%NORMALIZEDATA Summary of this function goes here
[M N D]=size(data);
data=reshape(data,[M*N D]);
data=scale_new(data);
data=reshape(data,[M N D]);
end
 function [data M m] =scale_new(data,M,m)
    [Nb_s Nb_b]=size(data);
    if nargin==1
        M=max(data,[],1);
        m=min(data,[],1);
    end
data = (data-repmat(m,Nb_s,1))./(repmat(M-m,Nb_s,1));
 end
%%
function [Isum,Igroupmax]=mI(img_gray)
img_gray=double(img_gray);
op3 = ones(3, 3);    op3(2, 2) = 0;
op5 = ones(5, 5);    op5(3, 3) = 0;
op7 = ones(7, 7);    op7(4, 4) = 0;
op9 = ones(9, 9);    op9(5, 5) = 0;
op11 = ones(11, 11);op11(6, 6) = 0;
op3 = op3 / sum(op3(:));
op5 = op5 / sum(op5(:));
op7 = op7 / sum(op7(:));
op9 = op9 / sum(op9(:));
op11 = op11 / sum(op11(:));
m3 = imfilter(img_gray, op3, 'symmetric');
m5 = imfilter(img_gray, op5, 'symmetric');
m7 = imfilter(img_gray, op7, 'symmetric');
m9 = imfilter(img_gray, op9, 'symmetric');
m11 = imfilter(img_gray, op11, 'symmetric');
Imap3 = img_gray - m3;
Imap5 = img_gray - m5;
Imap7 = img_gray - m7;
Imap9 = img_gray - m9;
Imap11 = img_gray - m11;
Imap3(Imap3<0) = 0;Imap5(Imap5<0) = 0;Imap7(Imap7<0) = 0;Imap9(Imap9<0) = 0;Imap11(Imap11<0) = 0;

Isum=Imap3+Imap5+Imap7+Imap9+Imap11;
igroup=cat(3,Imap3,Imap5,Imap7,Imap9,Imap11);
Igroupmax=max(igroup,[],3);
end
%% 局部梯度
function Ggroup = Gradmean(G1,G2,G3,G4,G5,G6,G7,G8,Nhood)
nhood=ones(Nhood);
tG1=G1;tG2=G2;tG3=G3;tG4=G4;tG5=G5;tG6=G6;tG7=G7;tG8=G8;

%--------对半方向——————————
nhood3=triu(nhood, 0);% 获取主对角线以上（含对角线）的元素，其余元素设置为0 
nhood4=flipud(nhood3);%上下镜像
nhood1=tril(nhood, 0);% 获取主对角线以下（含对角线）的元素，其余元素设置为0
nhood2=fliplr(nhood3);%左右镜像
nhood7=nhood; nhood7((Nhood+1)/2+1:end,:)=0;
nhood8=nhood; nhood8(:,1:(Nhood-1)/2)=0;
nhood5=flipud(nhood7); %上下镜像
nhood6=fliplr(nhood8);%左右镜像
%--------1/4方向——————————
nnhood1=nhood5.*nhood6;
nnhood2=nhood7.*nhood6;
nnhood3=nhood7.*nhood8;
nnhood4=nhood5.*nhood8;
nnhood5=nhood1.*nhood4;
nnhood6=nhood1.*nhood2;
nnhood7=nhood3.*nhood2;
nnhood8=nhood4.*nhood3;
G1sum=imfilter(tG1,nnhood1,'symmetric');G2sum=imfilter(tG2,nnhood2,'symmetric');
G3sum=imfilter(tG3,nnhood3,'symmetric');G4sum=imfilter(tG4,nnhood4,'symmetric');
G5sum=imfilter(tG5,nnhood5,'symmetric');G6sum=imfilter(tG6,nnhood6,'symmetric');
G7sum=imfilter(tG7,nnhood7,'symmetric');G8sum=imfilter(tG8,nnhood8,'symmetric');%计算非零总汇聚量
tG1(tG1>0)=1;tG2(tG2>0)=1;tG3(tG3>0)=1;tG4(tG4>0)=1;
tG5(tG5>0)=1;tG6(tG6>0)=1;tG7(tG7>0)=1;tG8(tG8>0)=1;
G1nsum=imfilter(tG1,nnhood1,'symmetric');G2nsum=imfilter(tG2,nnhood2,'symmetric');
G3nsum=imfilter(tG3,nnhood3,'symmetric');G4nsum=imfilter(tG4,nnhood4,'symmetric');
G5nsum=imfilter(tG5,nnhood5,'symmetric');G6nsum=imfilter(tG6,nnhood6,'symmetric');
G7nsum=imfilter(tG7,nnhood7,'symmetric');G8nsum=imfilter(tG8,nnhood8,'symmetric');%计算非零数量

G1mean=G1sum./G1nsum;
G2mean=G2sum./G2nsum;
G3mean=G3sum./G3nsum;
G4mean=G4sum./G4nsum;
G5mean=G5sum./G5nsum;
G6mean=G6sum./G6nsum;
G7mean=G7sum./G7nsum;
G8mean=G8sum./G8nsum;%计算非零元素平均值
G1mean(isnan(G1mean)) = 1;
G2mean(isnan(G2mean)) = 1;
G3mean(isnan(G3mean)) = 1;
G4mean(isnan(G4mean)) = 1;
G5mean(isnan(G5mean)) = 1;
G6mean(isnan(G6mean)) = 1;
G7mean(isnan(G7mean)) = 1;
G8mean(isnan(G8mean)) = 1;
Ggroup=cat(3, G1mean,G2mean,G3mean,G4mean,G5mean,G6mean,G7mean,G8mean);
end


function GgropM=LocalGrad(img_gray)
op1 =      [0,0,1;
            0,-1,0;
            0,0,0];
op2 =      [0,0,0;
            0,-1,0;
            0,0,1];
op3 =      [0,0,0;
            0,-1,0;
            1,0,0];
op4 =      [1,0,0;
            0,-1,0;
            0,0,0];
op5 =      [0,1,0;
            0,-1,0;
            0,0,0];
op6 =      [0,0,0;
            0,-1,1;
            0,0,0];
op7 =      [0,0,0;
            0,-1,0;
            0,1,0];
op8 =      [0,0,0;
            1,-1,0;
            0,0,0];
% We took 8 directions to simplify the calculation
G1=imfilter(img_gray, op1, 'replicate');
G2=imfilter(img_gray, op2, 'replicate');
G3=imfilter(img_gray, op3, 'replicate');
G4=imfilter(img_gray, op4, 'replicate');
G5=imfilter(img_gray, op5, 'replicate');
G6=imfilter(img_gray, op6, 'replicate');
G7=imfilter(img_gray, op7, 'replicate');
G8=imfilter(img_gray, op8, 'replicate');
G1(G1<0)=0;G2(G2<0)=0;G3(G3<0)=0;G4(G4<0)=0;
G5(G5<0)=0;G6(G6<0)=0;G7(G7<0)=0;G8(G8<0)=0;

Ggroup5 = Gradmean(G1,G2,G3,G4,G5,G6,G7,G8,5);
Ggroup9 = Gradmean(G1,G2,G3,G4,G5,G6,G7,G8,9);
Ggroup13 = Gradmean(G1,G2,G3,G4,G5,G6,G7,G8,13);
GgropM=cat(3,Ggroup5,Ggroup9,Ggroup13);
end


function oo = closeAP(fusion_map)
     se = strel('square',3);
    inputclose3 = imclose(fusion_map, se);
    inputclose3_u8 = uint8(mat2gray(inputclose3).*255);
    BW = imbinarize(inputclose3, 0.5*max(inputclose3(:)));

    stats = regionprops(BW, 'Area');
    if isempty(stats)
        sqrt_max = 5; % 默认值
    else
  % 计算尺寸分布权重（非简单中位数）
    areas = [stats.Area];
    max_size = max(areas);
    sqrt_max = sqrt(max_size);

    end
    diag_thresh = max(8, min(25, 2*ceil(sqrt_max))); % 阈值范围[8,25]

    inputAP2 = attribute_profile(inputclose3_u8, 'd', diag_thresh, 8, 'sub');
    oo = double(inputAP2(:,:,2) - inputAP2(:,:,3));
end

function num_peaks = estimatePeaks(diffImage)
% 步骤1: 预处理 - 降噪和增强对比度
    hsize = 3; % 高斯核的大小
    sigma = 1.0; % 高斯核的标准差
    gaussian_kernel = fspecial('gaussian', hsize, sigma);
    smoothed = imfilter(diffImage, gaussian_kernel, 'replicate'); % 使用'replicate'边界处理方式
    % figure; imshow(smoothed, []);title('smoothed');

    % 步骤2: 自适应阈值计算
    peakThreshold = 0.15 * max(smoothed(:)); % 基础阈值
    noiseLevel = median(smoothed(:));       % 噪声水平估计
    adaptiveThresh = max(peakThreshold, 1.5*noiseLevel); % 确保高于噪声 1.5

    binary_map = imbinarize(smoothed, adaptiveThresh);
    cleaned_bw = bwareaopen(binary_map, 2);

    cc = bwconncomp(cleaned_bw, 4); % 26-邻域连接
    num_peaks = cc.NumObjects;

end


end


