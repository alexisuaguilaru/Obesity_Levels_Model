import marimo

__generated_with = "0.14.9"
app = marimo.App(width="medium")


@app.cell
def _():
    # Importing auxiliar libraries
    import marimo as mo


    # Importing libraries

    import pandas as pd
    import numpy as np

    import seaborn as sns
    import matplotlib.pyplot as plt


    # Importing Functions and Utils

    import SourceExploratoryDataAnalysis as src
    return mo, pd, sns, src


@app.cell
def _():
    # Defining useful variables

    PATH = './Datasets/'
    PATH_DATASET = PATH + 'ObesityDataset{}.csv'
    RANDOM_STATE = 8013
    return PATH_DATASET, RANDOM_STATE


@app.cell
def _(mo):
    mo.md(r"# Exploratory Data Analysis")
    return


@app.cell
def _(mo):
    mo.md(r"")
    return


@app.cell
def _(mo):
    mo.md(r"# 1. First Exploration")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        The dataset consists of $2111$ instances, of which the following attributes are reported:
    
        * `Gender`
        * `Age`
        * `Height`
        * `Weight`
        * `family_history_with_overweight`: Has a family member suffered or suffers from overweight?
        * `FAVC`: Do you eat high caloric food frequently?
        * `FCVC`: Do you usually eat vegetables in your meals?
        * `NCP`: How many main meals do you have daily?
        * `CAEC`: Do you eat any food between meals?
        * `SMOKE`: Do you smoke?
        * `CH2O`: How much water do you drink daily?
        * `SCC`: Do you monitor the calories you eat daily?
        * `FAF`: How often do you have physical activity?
        * `TUE`: How much time do you use technological devices such as cell phone, videogames, television, computer and others?
        * `CALC`: How often do you drink alcohol?
        * `MTRANS`: Which transportation do you usually use?
        * `NObeyesdad`: Obesity level
    
        The last attribute is the one to be predicted with the different models to be defined and which is distributed as follows according to the values it takes:
    
        * `Insufficient_Weight`: $272$
        * `Normal_Weight`: $287$
        * `Overweight_Level_I`: $290$
        * `Overweight_Level_II`: $290$
        * `Obesity_Type_I`: $351$
        * `Obesity_Type_II`: $297$
        * `Obesity_Type_III`: $324$
    
        It can be observed that there is an imbalance between the classes, so this could cause difficulties when training the model to be able to generate a good separation or classification of the instances. There are no missing values, so we can proceed directly with the analysis.
        """
    )
    return


@app.cell
def _(PATH_DATASET, pd):
    # Loading dataset

    ObesityDataset_Raw_0 = pd.read_csv(PATH_DATASET.format(''),engine='python')
    return (ObesityDataset_Raw_0,)


@app.cell
def _(ObesityDataset_Raw_0, src):
    # Splitting features into Numerical, Categorical and Target features

    NumericalFeatures , CategoricalFeatures , ObesityLevel = src.SplitFeatures(ObesityDataset_Raw_0)
    return CategoricalFeatures, NumericalFeatures, ObesityLevel


@app.cell
def _(ObesityDataset_Raw_0, ObesityLevel, RANDOM_STATE, mo):
    _Sample = ObesityDataset_Raw_0.groupby(ObesityLevel).sample(2,random_state=RANDOM_STATE)
    mo.vstack(
        [
            mo.md("**Examples of Instances**"),
            _Sample,
        ]
    )
    return


@app.cell
def _(ObesityDataset_Raw_0, mo):
    mo.vstack(
        [
            mo.md("**Data types of Features**"),
            ObesityDataset_Raw_0.dtypes,
        ]
    )
    return


@app.cell
def _(ObesityDataset_Raw_0, mo):
    mo.vstack(
        [
            mo.md(r"Dataset Contains **$0$ Missing Values**"),
            ObesityDataset_Raw_0.isnull().sum(),
        ]
    )
    return


@app.cell
def _(CategoricalFeatures, ObesityDataset_Raw_0, src):
    # Capitalizing of `yes` and `no` values

    ObesityDataset_Raw_1 = ObesityDataset_Raw_0.copy(deep=True)
    ObesityDataset_Raw_1[CategoricalFeatures] = ObesityDataset_Raw_1[CategoricalFeatures].map(src.CapitalizeYesNoValues)
    return (ObesityDataset_Raw_1,)


@app.cell
def _(CategoricalFeatures, ObesityDataset_Raw_1, ObesityLevel, mo):
    mo.vstack(
        [
            mo.md("Number of Instances by **Obesity Level**"),
            ObesityDataset_Raw_1.groupby(ObesityLevel)[CategoricalFeatures[0]].count(),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        When observing the values taken by some of the categorical features, it can be determined that they have ordinal behavior (their values are frequencies) or binary (their values are `yes` or `no`); therefore, they can be encoded without losing their sense or meaning. In addition, the feature that represents the target, `NObeyesdad`, can also be associated with an order because its values represent how obesity increases and scales in an individual.
    
        Finally, the feature `IBM` (Body Mass Index) is added, which represents one of the main and most widely used measures to describe health condition and is directly related to the level of obesity of a person.
        """
    )
    return


@app.cell
def _():
    # Defining auxiliar variables (feature names)

    ## Numerical features
    Age = 'Age'
    Height = 'Height'
    Weight = 'Weight'
    FCVC = 'FCVC'
    NCP = 'NCP'
    CH2O = 'CH2O'
    FAF = 'FAF'
    TUE = 'TUE'

    ## Nominal features
    Gender = 'Gender'
    MTRANS = 'MTRANS'

    ## Binary features
    FamilyOverweight = 'family_history_with_overweight'
    FAVC = 'FAVC'
    SMOKE = 'SMOKE'
    SCC = 'SCC'

    ## Frequency features
    CAEC = 'CAEC'
    CALC = 'CALC'
    return CAEC, CALC, FAVC, FamilyOverweight, Height, SCC, SMOKE, Weight


@app.cell
def _(
    CAEC,
    CALC,
    FAVC,
    FamilyOverweight,
    Height,
    ObesityDataset_Raw_1,
    ObesityLevel,
    SCC,
    SMOKE,
    Weight,
    src,
):
    ObesityDataset_1 = ObesityDataset_Raw_1.copy()


    # Feature Encoding 
    _BinFeatures = [FamilyOverweight,FAVC,SMOKE,SCC]
    ObesityDataset_1[_BinFeatures] = ObesityDataset_1[_BinFeatures].map(src.EncodeBinaryValue)

    _FreqFeatures = [CAEC,CALC]
    ObesityDataset_1[_FreqFeatures] = ObesityDataset_1[_FreqFeatures].map(src.EncodeFrequencyValue)

    ObesityDataset_1[[ObesityLevel]] = ObesityDataset_1[[ObesityLevel]].map(src.EncodeObesityLevel)

    # Adding feature 
    BMI = 'BMI'
    ObesityDataset_1[BMI] = ObesityDataset_1[Weight] / (ObesityDataset_1[Height]**2)
    return (ObesityDataset_1,)


@app.cell
def _(ObesityDataset_1, ObesityLevel, RANDOM_STATE, mo):
    _Sample = ObesityDataset_1.groupby(ObesityLevel).sample(2,random_state=RANDOM_STATE)
    mo.vstack(
        [
            mo.md("**Examples of Instances After Transformation**"),
            _Sample,
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"# Data Cleaning [I]")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        The dataset was loaded and the features were classified into numerical, categorical and target type. The categorical features were encoded based on meaning of their values, in other words what represent. 
    
        The full dataset was grouped by insufficient-normal weight, overweight and obesity because the final model will firstly estimate the probabilities of belonging to these groups and then those belonging to a certain obesity level.
        """
    )
    return


@app.cell
def _(pd):
    # Loading raw dataset
    ObesityDataset_Raw = pd.read_csv('ObesityDataset_Raw.csv')
    return (ObesityDataset_Raw,)


@app.cell
def _(ObesityDataset_Raw):
    # Getting categorical and numerical features
    CategoricalFeatures = []
    NumericalFeatures = []
    TargetFeature = ['NObeyesdad']

    for label in ObesityDataset_Raw.columns:
        if label not in TargetFeature:
            if ObesityDataset_Raw[label].dtype == 'object':
                CategoricalFeatures.append(label)
            else:
                NumericalFeatures.append(label)
    return CategoricalFeatures, NumericalFeatures, TargetFeature


@app.cell
def _(
    CapitalizeIfNecessary,
    CategoricalFeatures,
    ObesityDataset_Raw,
    TargetFeature,
):
    # Capitalizing values of categorical features
    ObesityDataset_Raw[CategoricalFeatures+TargetFeature] = ObesityDataset_Raw[CategoricalFeatures+TargetFeature].map(CapitalizeIfNecessary)
    return


@app.cell
def _(CategoricalFeatures, ObesityDataset_Raw, TargetFeature):
    # Transforming categorical features' data type into categorical data type
    ObesityDataset_Raw[CategoricalFeatures+TargetFeature] = ObesityDataset_Raw[CategoricalFeatures+TargetFeature].astype('category')
    return


@app.cell
def _(CategoricalFeatures, EncodeCategoricalFeature, ObesityDataset_Raw):
    SortedValues_NObeyesdad = ['Insufficient_Weight', 'Normal_Weight', 'Overweight_Level_I', 'Overweight_Level_II', 'Obesity_Type_I', 'Obesity_Type_II', 'Obesity_Type_III']
    EncodeCategoricalFeature(ObesityDataset_Raw, 'NObeyesdad', SortedValues_NObeyesdad)
    Values_Gender = ['Female', 'Male']
    EncodeCategoricalFeature(ObesityDataset_Raw, 'Gender', Values_Gender)
    Values_Frequencies = ['No', 'Sometimes', 'Frequently', 'Always']
    CategoricalFeatures_Frequencies = ['CAEC', 'CALC']
    for _feature in CategoricalFeatures_Frequencies:
        EncodeCategoricalFeature(ObesityDataset_Raw, _feature, Values_Frequencies)
    Values_MTRANS = ['Walking', 'Bike', 'Public_Transportation', 'Motorbike', 'Automobile']
    EncodeCategoricalFeature(ObesityDataset_Raw, 'MTRANS', Values_MTRANS)
    Values_Binary = ['No', 'Yes']
    EncodedFeatures = set(['Gender', 'MTRANS'] + CategoricalFeatures_Frequencies)
    for _feature in CategoricalFeatures:
        if _feature not in EncodedFeatures:
            EncodeCategoricalFeature(ObesityDataset_Raw, _feature, Values_Binary)
    return


@app.cell
def _(CategoricalFeatures):
    CategoricalFeatures_Encode = [_feature + '_Encode' for _feature in CategoricalFeatures]
    TargetFeature_Encode = ['NObeyesdad_Encode']
    return CategoricalFeatures_Encode, TargetFeature_Encode


@app.cell
def _(ObesityDataset_Raw, SegmentingDatasetByCategories, TargetFeature):
    # Grouping by Obesity level (NObeyesdad)
    GroupObesityLevel = ObesityDataset_Raw.groupby(by=TargetFeature,observed=True)

    # Segmenting Dataset by Obesity levels
    ObesityLevels = dict()
    for obesity_level in ObesityDataset_Raw['NObeyesdad'].unique():
        ObesityLevels[obesity_level] = GroupObesityLevel.get_group((obesity_level,))

    ## Insufficient and Normal Weight
    ObesityLevels_Group1 = ['Insufficient_Weight','Normal_Weight']
    ObesityDataset_Group1 = SegmentingDatasetByCategories(ObesityLevels,'NObeyesdad',ObesityLevels_Group1)

    ## Overweight 
    ObesityLevels_Group2 = ['Overweight_Level_I','Overweight_Level_II']
    ObesityDataset_Group2 = SegmentingDatasetByCategories(ObesityLevels,'NObeyesdad',ObesityLevels_Group2)

    ## Obesity
    ObesityLevels_Group3 = ['Obesity_Type_I','Obesity_Type_II','Obesity_Type_III']
    ObesityDataset_Group3 = SegmentingDatasetByCategories(ObesityLevels,'NObeyesdad',ObesityLevels_Group3)
    return (
        ObesityDataset_Group1,
        ObesityDataset_Group2,
        ObesityDataset_Group3,
        ObesityLevels_Group1,
        ObesityLevels_Group2,
        ObesityLevels_Group3,
    )


@app.cell
def _(
    ObesityDataset_Raw,
    ObesityLevels_Group1,
    ObesityLevels_Group2,
    ObesityLevels_Group3,
):
    # Re-encoding Obesity Level based on groups
    ObesityDataset_Groups = ObesityDataset_Raw.copy(deep=True)
    Encoder_ObesityLevelGroups = {obesity_level:group_num for group_num , group_ObesityLevel in enumerate([ObesityLevels_Group1,ObesityLevels_Group2,ObesityLevels_Group3]) for obesity_level in group_ObesityLevel}
    ObesityDataset_Groups['NObeyesdad_Encode'] = ObesityDataset_Groups['NObeyesdad'].map(Encoder_ObesityLevelGroups).astype(int)

    # Grouping obesity levels
    ObesityLevelGroups = ['Insufficient-Normal Weight','Overweight', 'Obesity']
    Decoder_ObesityLevelGroups = {group_num:group_ObesityLevel for group_num , group_ObesityLevel in enumerate(ObesityLevelGroups)}
    ObesityDataset_Groups['NObeyesdad'] = ObesityDataset_Groups['NObeyesdad_Encode'].map(Decoder_ObesityLevelGroups).astype('category')
    return ObesityDataset_Groups, ObesityLevelGroups


@app.cell
def _(mo):
    mo.md(r"# Data Analysis")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        In the full dataset, the amount of records with the different combinations of Obesity Level and Gender were counted and the features' distribution were plotted to show their possible distributions. 
    
        It was plotted the correlation matrix in each sub dataset [by Groups, Group 1, Group 2 and Group 3] to show lineal relation between features and the target and some observations were made about the interaction between features and mainly with the target.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(r"## Full Dataset")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        Wether the records in the dataset are counted by obesity level and gender, can be seen that the number of records in some pair of values is almost nil compared to others. For example, in obesity type II and obesity type III is observed this situation.
    
        Therefore obesity type II, female and obesity type III, male are clases with low representation, so (preliminarily) the final model will not be able to learn enough to estimate probabilities in samples with this features and categories. But in the others will not apparently show this type of difficult.
        """
    )
    return


@app.cell
def _(ObesityDataset_Raw):
    # Counting records by Obesity Level and Gender
    ObesityDataset_Raw[['NObeyesdad','Gender']].value_counts()
    return


@app.cell
def _(mo):
    mo.md(r"Some of the features do not have a clear distribution except age, height and weight. The former is like a $\chi^2$ distribution and the last are like normal distribution. And the target feature seems like a normal distribution which tend to be uniform.")
    return


@app.cell
def _(NumericalFeatures, ObesityDataset_Raw, PlotHistogramBox):
    for _feature in NumericalFeatures:
        PlotHistogramBox(ObesityDataset_Raw, _feature, 'Numerical')
    return


@app.cell
def _(
    CategoricalFeatures,
    ObesityDataset_Raw,
    PlotHistogramBox,
    TargetFeature,
):
    for _feature in CategoricalFeatures + TargetFeature:
        PlotHistogramBox(ObesityDataset_Raw, _feature, 'Categorical')
    return


@app.cell
def _(mo):
    mo.md(r"## Dataset by Groups")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        In the dataset by groups, age, weight, family history with overweight, FAVC (eat high caloric food frequently) and CAEC (eat any food between meals) are feature that are more correlated with the target feature. The weight having the greatest impact on predicting the target's value.
    
        Other features do not show a hight correlation value with the target features, implies that either are independent variables or their interaction is no lineal. 
    
        Another observation to be made is that the another features related to habits may not be relevant to classify a data point into Insufficient-Normal Weight, Overweight or Obesity [class for the first sub-model] but be to distinguish between obesity levels.
        """
    )
    return


@app.cell
def _(
    CategoricalFeatures,
    CategoricalFeatures_Encode,
    NumericalFeatures,
    ObesityDataset_Groups,
    PlotHeatmapCorrelation,
    TargetFeature,
    TargetFeature_Encode,
):
    # Plotting correlation between features in grouped dataset
    Features_Encode = NumericalFeatures+CategoricalFeatures_Encode+TargetFeature_Encode
    TickLabels_Y = NumericalFeatures+CategoricalFeatures+TargetFeature
    PlotHeatmapCorrelation(ObesityDataset_Groups,Features_Encode,TickLabels_Y)
    return Features_Encode, TickLabels_Y


@app.cell
def _(mo):
    mo.md(r"The distribution of some features with a high correlation with the target, show that certain values in these features allow distinguish between groups. Therefor certain values will determine to which group belongs.")
    return


@app.cell
def _(ObesityDataset_Groups, ObesityLevelGroups, PlotHistogramBox_Hue):
    PlotHistogramBox_Hue(ObesityDataset_Groups,'Weight','NObeyesdad',ObesityLevelGroups)
    return


@app.cell
def _(ObesityDataset_Groups, ObesityLevelGroups, PlotHistogramBox_Hue):
    PlotHistogramBox_Hue(ObesityDataset_Groups,'Age','NObeyesdad',ObesityLevelGroups)
    return


@app.cell
def _(ObesityDataset_Groups, ObesityLevelGroups, sns):
    sns.histplot(ObesityDataset_Groups,x='family_history_with_overweight',hue='NObeyesdad',hue_order=ObesityLevelGroups,multiple='stack',palette='Set2')
    return


@app.cell
def _(ObesityDataset_Groups, ObesityLevelGroups, sns):
    sns.histplot(ObesityDataset_Groups,x='FAVC',hue='NObeyesdad',hue_order=ObesityLevelGroups,multiple='stack',palette='Set2')
    return


@app.cell
def _(ObesityDataset_Groups, ObesityLevelGroups, sns):
    sns.histplot(ObesityDataset_Groups,x='CAEC',hue='NObeyesdad',hue_order=ObesityLevelGroups,multiple='stack',palette='Set2')
    return


@app.cell
def _(ObesityDataset_Groups, sns):
    sns.histplot(ObesityDataset_Groups,x='NObeyesdad')
    return


@app.cell
def _(mo):
    mo.md(r"## Dataset Group 1 :: Insufficient - Normal Weight")
    return


@app.cell
def _(mo):
    mo.md(r"In the datasets group 1, the features related to habits do not have a high correlation with the target, but with others features. This mean that these features have a no lineal relation with the target.")
    return


@app.cell
def _(
    Features_Encode,
    ObesityDataset_Group1,
    PlotHeatmapCorrelation,
    TickLabels_Y,
):
    # Plotting correlation between features in group 1 :: Insufficient and Normal weight
    PlotHeatmapCorrelation(ObesityDataset_Group1,Features_Encode,TickLabels_Y)
    return


@app.cell
def _(mo):
    mo.md(r"The plots of weight versus some features with low correlation with the target show how certain combinations of values allow distinguish between Insufficient and Normal.")
    return


@app.cell
def _(ObesityDataset_Group1, sns):
    sns.scatterplot(ObesityDataset_Group1,x='Weight',y='NCP',hue='NObeyesdad',palette='Set2')
    return


@app.cell
def _(ObesityDataset_Group1, sns):
    sns.scatterplot(ObesityDataset_Group1,x='Weight',y='CH2O',hue='NObeyesdad',palette='Set2')
    return


@app.cell
def _(ObesityDataset_Group1, sns):
    sns.scatterplot(ObesityDataset_Group1,x='Weight',y='family_history_with_overweight',hue='NObeyesdad',palette='Set2')
    return


@app.cell
def _(ObesityDataset_Group1, sns):
    sns.scatterplot(ObesityDataset_Group1,x='Weight',y='SCC',hue='NObeyesdad',palette='Set2')
    return


@app.cell
def _(mo):
    mo.md(r"## Dataset Group 2 :: Overweight")
    return


@app.cell
def _(mo):
    mo.md(r"In the datasets group 2, the features related to habits have a bit higher correlation with the target, and also with others features. This mean that these features tend to have a lineal relation with the target.")
    return


@app.cell
def _(
    Features_Encode,
    ObesityDataset_Group2,
    PlotHeatmapCorrelation,
    TickLabels_Y,
):
    # Plotting correlation between features in group 2 :: Overweight
    PlotHeatmapCorrelation(ObesityDataset_Group2,Features_Encode,TickLabels_Y)
    return


@app.cell
def _(mo):
    mo.md(r"The plots of weight and age versus some features with high correlation with the target show how certain combinations of values allow distinguish between Overweight Level I and II. Also show a preference, by way of distribution, how this happens.")
    return


@app.cell
def _(ObesityDataset_Group2, sns):
    sns.scatterplot(ObesityDataset_Group2,x='Weight',y='SCC',hue='NObeyesdad',palette='Set2')
    return


@app.cell
def _(ObesityDataset_Group2, sns):
    sns.scatterplot(ObesityDataset_Group2,x='Weight',y='CALC',hue='NObeyesdad',palette='Set2')
    return


@app.cell
def _(ObesityDataset_Group2, sns):
    sns.scatterplot(ObesityDataset_Group2,x='Weight',y='family_history_with_overweight',hue='NObeyesdad',palette='Set2')
    return


@app.cell
def _(ObesityDataset_Group2, sns):
    sns.scatterplot(ObesityDataset_Group2,x='Age',y='CAEC',hue='NObeyesdad',palette='Set2')
    return


@app.cell
def _(mo):
    mo.md(r"## Dataset Group 3 :: Obesity")
    return


@app.cell
def _(mo):
    mo.md(r"In the datasets group 3, the features related to habits have a higher correlation with the others features (including the target). This mean that the features are more sensitive for predicting the target.")
    return


@app.cell
def _(
    Features_Encode,
    ObesityDataset_Group3,
    PlotHeatmapCorrelation,
    TickLabels_Y,
):
    # Plotting correlation between features in group 3 :: Obesity
    PlotHeatmapCorrelation(ObesityDataset_Group3,Features_Encode,TickLabels_Y)
    return


@app.cell
def _(mo):
    mo.md(r"The plots of weight versus some features with high correlation with the target show patterns between the target and certain combinations of values. Therefore, these values allow distinguish between Obesity Type I, II and III.")
    return


@app.cell
def _(ObesityDataset_Group3, sns):
    sns.scatterplot(ObesityDataset_Group3,x='Weight',y='FCVC',hue='NObeyesdad',palette='Set2')
    return


@app.cell
def _(ObesityDataset_Group3, sns):
    sns.scatterplot(ObesityDataset_Group3,x='Weight',y='NCP',hue='NObeyesdad',palette='Set2')
    return


@app.cell
def _(ObesityDataset_Group3, sns):
    sns.scatterplot(ObesityDataset_Group3,x='Weight',y='Gender',hue='NObeyesdad',palette='Set2')
    return


@app.cell
def _(mo):
    mo.md(r"# Data Cleaning [II]")
    return


@app.cell
def _(mo):
    mo.md(r"The full and sub datasets with the encoded and numerical features were saved as CSV file. It was not necessary to eliminate the outliers because they contain useful information for a proper classification and estimation.")
    return


@app.cell
def _(
    CategoricalFeatures,
    CategoricalFeatures_Encode,
    NumericalFeatures,
    ObesityDataset_Group1,
    ObesityDataset_Group2,
    ObesityDataset_Group3,
    ObesityDataset_Groups,
    ObesityDataset_Raw,
    RenameColumnLabels,
    TargetFeature,
    TargetFeature_Encode,
):
    # Save Full and Sub Datasets as CSV files
    FeaturesSubset = NumericalFeatures + CategoricalFeatures_Encode + TargetFeature_Encode

    ApplyRenameLabels = RenameColumnLabels(FeaturesSubset,NumericalFeatures+CategoricalFeatures+TargetFeature)

    ApplyRenameLabels(ObesityDataset_Raw).to_csv('./ObesityDataset_Clean.csv')
    ApplyRenameLabels(ObesityDataset_Groups).to_csv('./ObesityLevel_Groups.csv')
    ApplyRenameLabels(ObesityDataset_Group1).to_csv('./ObesityLevel_Group1.csv')
    ApplyRenameLabels(ObesityDataset_Group2).to_csv('./ObesityLevel_Group2.csv')
    ApplyRenameLabels(ObesityDataset_Group3).to_csv('./ObesityLevel_Group3.csv')
    return


if __name__ == "__main__":
    app.run()
