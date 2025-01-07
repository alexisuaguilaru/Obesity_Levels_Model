import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def CapitalizeIfNecessary(value:str):
    """
        Function to capitalize 'no' and 'yes' 
        strings
    """
    if value == 'no' or value == 'yes':
        return value.capitalize()
    else:
        return value
    
def EncodeCategoricalFeature(Dataset:pd.DataFrame,CategoricalFeature:str,SortedValues_Feature:list[str]):
    """
        Function to encode a categorical feature with a given 
        sorted list of values
    """
    Encoder_Feature = {category_value:value_encode for value_encode , category_value in enumerate(SortedValues_Feature)}
    Dataset[CategoricalFeature+'_Encode'] = Dataset[CategoricalFeature].map(Encoder_Feature).astype(int)

def SegmentingDatasetByCategories(SegmentDatasets:dict[str,pd.DataFrame],SegmentedFeature:str,GroupCategories:list[str]):
    """
        Function to segment a dataset based on a group of values in a feature
    """
    SegmentedDataset = pd.concat([SegmentDatasets[category] for category in GroupCategories])
    SegmentedDataset[SegmentedFeature] = SegmentedDataset[SegmentedFeature].astype('object').astype('category')
    return SegmentedDataset

def PlotHistogramBox(Dataset:pd.DataFrame,Feature:str,TypeFeature:str):
    """
        Function for plotting histogram along with boxplot (if feature is numerical)
    """
    if TypeFeature == 'Numerical':
        fig , axes = plt.subplots(1,2,figsize=(15,5))
        sns.histplot(Dataset,x=Feature,ax=axes[0])
        sns.boxplot(Dataset,x=Feature,ax=axes[1])
    elif TypeFeature == 'Categorical':
        fig , axes = plt.subplots(figsize=(15,5))
        sns.histplot(Dataset,x=Feature,ax=axes)

def PlotHeatmapCorrelation(Dataset:pd.DataFrame,Features_Encode:list[str],Features_Names:list[str],Threshold:float=0.1):
    """
        Function to plot correlation matrix of a dataset with a given features
    """
    CorrelationMatriz = Dataset[Features_Encode].corr()
    MaskValues = CorrelationMatriz.apply(lambda value: abs(value)<=Threshold)
    sns.heatmap(CorrelationMatriz,mask=MaskValues,cmap='RdYlBu',yticklabels=Features_Names,xticklabels=False,vmax=1,vmin=-1,annot=True,annot_kws={'size':6})

def PlotHistogramBox_Hue(Dataset:pd.DataFrame,Feature:str,Hue:str,Hue_Order=list[str]):
    """
        Function for plotting histogram along with boxplot with group of values (hue)
    """
    fig , axes = plt.subplots(1,2,figsize=(15,5))
    sns.kdeplot(Dataset,x=Feature,ax=axes[0],hue=Hue,hue_order=Hue_Order,fill=True,palette='Set2')
    sns.boxplot(Dataset,x=Feature,ax=axes[1],hue=Hue,hue_order=Hue_Order,palette='Set2')

def RenameColumnLabels(OldLabels:list[str],NewLabels:list[str]):
    """
        Function to rename a subset of columns labels and to return 
        the new dataframe
    """
    mapFeatures = dict(zip(OldLabels,NewLabels))
    def ApplyRename(Dataset:pd.DataFrame):
        datasetRelabeled = Dataset[OldLabels].copy(deep=True)
        return datasetRelabeled.rename(columns=mapFeatures)

    return ApplyRename