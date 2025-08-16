function [ Iscore, t_real ] = iNNE( traindata,testdata,t,psi,m,n )

rng('shuffle','multFibonacci')
index=cell(t,1);
for i=1:t
    index{i} = iNNE_Cigrid( traindata,psi ,m,n,i);
end

n=size(testdata,1);
Iso=zeros(n,t)+1;

for i=1:t
    pindex=index{i}(1,:);
    distIndex=index{i}(3,:);
    ratioindex=index{i}(end,:);    
    pdata=traindata(pindex,:);

 %--------------------------------------------------%
    [D] = pdist2(pdata,testdata,'minkowski',2);
    radiusMatrix=repmat(distIndex',1,n);
    I=D<radiusMatrix; % find balls covering x
    Cradius=radiusMatrix.*I;
    Cradius(Cradius==0)=1;
    [~,II]=min(Cradius,[],1); % find cnn(x) 
    Cratio=ratioindex(II);
    
    Cratio(sum(I,1)==0)=1;
    Iso(:,i)=Cratio;
    t_real=i;
    
    
    if i>30 %´Ó30¿ªÊ¼
        temp1=Iso(:,1:i-1);tempIso1=mean(temp1,2);stdIso1=std(tempIso1);meanIso1=mean(tempIso1);
        temp2=Iso(:,1:i);tempIso2=mean(temp2,2);stdIso2=std(tempIso2);meanIso2=mean(tempIso2);
        Iscore=tempIso2;
        if (abs(stdIso1-stdIso2)<0.0001 && abs(meanIso1-meanIso2)<0.005)
            break;
        end
    end

  
end

% Iscore=mean(Iso,2);

    
