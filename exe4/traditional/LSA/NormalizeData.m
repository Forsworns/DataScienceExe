function Data = NormalizeData(Data)
    Data = Data ./ repmat(sum(Data,2),1,size(Data,2)); 
    Data = zscore(Data,1);  
end