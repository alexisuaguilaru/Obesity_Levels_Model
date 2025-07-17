import marimo

__generated_with = "0.14.10"
app = marimo.App(width="medium")


@app.cell
def _():
    # Importing auxiliar libraries
    import marimo as mo


    # Importing libraries

    import pandas as pd
    import numpy as np
    from scipy import stats
    from statsmodels.multivariate.factor import Factor

    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.decomposition import PCA , KernelPCA
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    import seaborn as sns
    import matplotlib.pyplot as plt


    # Importing Functions and Utils

    import SourceExploratoryDataAnalysis as src
    return (
        ColumnTransformer,
        Factor,
        KMeans,
        MinMaxScaler,
        PCA,
        Pipeline,
        mo,
        np,
        pd,
        silhouette_score,
        src,
        stats,
    )


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
    ObesityDataset_Raw_0.columns = ObesityDataset_Raw_0.columns.astype(str)
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
    return (
        CAEC,
        CALC,
        FAF,
        FAVC,
        FamilyOverweight,
        Gender,
        Height,
        MTRANS,
        SCC,
        SMOKE,
        Weight,
    )


@app.cell
def _(
    CAEC,
    CALC,
    FAVC,
    FamilyOverweight,
    Height,
    NumericalFeatures,
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
    NumericalFeatures.append(BMI)
    ObesityDataset_1[BMI] = ObesityDataset_1[Weight] / (ObesityDataset_1[Height]**2)
    return BMI, ObesityDataset_1


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
    mo.md(r"# 2. Data Analysis")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        This section analyzes the impact of certain factors (attributes) on the level of obesity, where it is shown that BMI is the main attribute that determines the obesity status of a person and that together with other factors, the obesity status of a person could be determined with greater precision.
    
        From the analysis carried out, it can be synthesized that an individual's obesity is mainly determined by the actions and decisions he/she makes, as well as by the influence of his/her environment (such as access to ultra-processed products and overweight family members); therefore, obesity is considered to be a multifactorial disease where the interaction of several circumstances leads to the manifestation of a certain state of health in a person.
    
        Although not all the other factors were analyzed, it could be suggested that their influence is less or equal to the factors considered here, because, as mentioned, BMI is the attribute that has the highest weighting when rating the level of obesity in a person (this is shown at a practical level, as it is widely used by doctors, as well as by the statistical analysis performed).
        """
    )
    return


@app.cell
def _(mo):
    mo.md(r"## 2.1.Body Mass Index (BMI)")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        As mentioned, the [Body Mass Index (BMI)](https://en.wikipedia.org/wiki/Body_mass_index) is a metric that allows to sufficiently describe the physical condition of a person; this is directly related to the level and state of his or her obesity. Therefore, a range of values that the BMI takes to determine whether a person is at normal weight, overweight or obese has been proposed. By observing the plot generated, it can be determined that these ranges of values exist although they are not completely accurate.
    
        As the complexity and level of obesity increases so do the mean and median BMI, this represents a statistically strong correlation ($\rho = 0.9778$) and association between these two measures. In addition, the range formed by the first and third quartiles for the different levels of obesity do not overlap, so the values will present significant differences.
    
        From the above, it could be concluded that the `IBM` of a person can generate a good first estimate of the obesity suffered by an individual, reason enough to use it as a preferred metric to describe the physical condition. But to improve its accuracy, factors related to physical activity and nutrition (diet) could be considered.
        """
    )
    return


@app.cell
def _(BMI, ObesityDataset_1, src):
    src.SummaryStatisticsFeature(
        ObesityDataset_1,
        BMI,
    )
    return


@app.cell
def _(BMI, ObesityDataset_1, ObesityDataset_Raw_1, ObesityLevel, src):
    src.PlotFeatureOverCategories(
        ObesityDataset_1,
        BMI,
        ObesityDataset_Raw_1[ObesityLevel],
        FeatureName='Body Mass Index',
        CategoryName='Obesity Level',
    )
    return


@app.cell
def _(mo):
    mo.md(r"Applying the [t-test](https://en.wikipedia.org/wiki/Student%27s_t-test) on consecutive levels of obesity shows that there is a significant difference in the means of `BMI` with a significance level of $\alpha = 0.01$, thus showing that `BMI` is a differentiating metric for determining the overall health status of an individual.")
    return


@app.cell
def _(BMI, ObesityDataset_1, ObesityLevel, src, stats):
    # Applying t-test for difference between means

    for _level_1 in range(6):
        _level_2 = _level_1 + 1
        _Sample_1 = ObesityDataset_1.query(f'{ObesityLevel} == {_level_1}')[BMI]
        _Sample_2 = ObesityDataset_1.query(f'{ObesityLevel} == {_level_2}')[BMI]
        _result = stats.ttest_ind(
            _Sample_1,
            _Sample_2,
            equal_var=False,
            alternative='less',
        )
        print(f'{src.MapLevelObesity[_level_1]:<20} < {src.MapLevelObesity[_level_2]:<20} :: {'Yes' if _result.pvalue < 0.01 else 'No'}')
    return


@app.cell
def _(mo):
    mo.md(r"## 2.2. Family History With Overweight")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        A fact that is common when a patient suffers from obesity is that the patient's family circle also suffers from it, this can be shown by observing that there is a positive trend between the complexity of obesity and the number of cases where they have a family member with obesity. This fact becomes evident when considering the plot, where two considerable biases are appreciated, that is, a patient who does not have overweight relatives tends not to have a high level of obesity and if he/she does, he/she tends to have a high level of obesity.
    
        The above observations allow to describe `family_history_with_overweight` as a risk factor that does not have a direct impact or strong influence, but rather allows to observe the health condition of the family circle and what one might observe in it.
        """
    )
    return


@app.cell
def _(FamilyOverweight, ObesityDataset_1, src):
    PivotFamilyOverweight = src.SummaryCategoricalFeature(
        ObesityDataset_1,
        FamilyOverweight,
    )

    PivotFamilyOverweight
    return (PivotFamilyOverweight,)


@app.cell
def _(PivotFamilyOverweight, src):
    src.PlotPivotTable(
        PivotFamilyOverweight,
        'Relatives\nwith Overweight',
    )
    return


@app.cell
def _(mo):
    mo.md(r"## 2.3. Physical Activity")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        The frequency and the amount of hours of physical activity become factors that allow to combat and reduce obesity in a person, therefore it becomes a habit that is constantly promoted. To show how physical activity influences the level of obesity, a decrease should be observed as obesity increases.
    
        When considering the medians in the different levels, it can be appreciated that there is no uniform or noticeable decrease, only in the extreme levels these decrements can be appreciated. But there is a bias that allows a better evaluation of this phenomenon, most of the levels (except for Obesity Type III) have a negative bias, this could be an indication of a promotion of physical activity among these groups and that part of the population tries to perform this physical activation. This impacts on a reduction of obesity over time and, in general, a concern or interest in having a healthy life.
    
        Although there is a tendency to do physical activity, there is still a high variance in each group, which means that not all individuals or patients follow the suggestion to do physical activity, so there is the possibility of encouraging it and increasing the time invested in health care.
        """
    )
    return


@app.cell
def _(FAF, ObesityDataset_1, src):
    src.SummaryStatisticsFeature(
        ObesityDataset_1,
        FAF,
    )
    return


@app.cell
def _(FAF, ObesityDataset_1, ObesityDataset_Raw_1, ObesityLevel, src):
    src.PlotFeatureOverCategories(
        ObesityDataset_1,
        FAF,
        ObesityDataset_Raw_1[ObesityLevel],
        FeatureName='Frequency of\nPhysical Activity',
        CategoryName='Obesity Level',
    )
    return


@app.cell
def _(mo):
    mo.md(r"Omitting the extreme cases (Insufficient Weight and Obesity Type III) that follow the trend of decreasing physical activity as obesity increases, there remain the levels where apparently no clear trend is shown. Using the [Spearman Rank-Order Correlation Coefficient](https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient) it can be shown that in these levels the correlation is not strong although its p-value is significant; this could indicate that the effect of physical activity is not appreciable, that is, there is a similar number of cases that follow the trend as those that do not follow it and the overall effect becomes negligible.")
    return


@app.cell
def _(FAF, ObesityDataset_1, ObesityLevel, stats):
    _Data = ObesityDataset_1[[FAF,ObesityLevel]].query(f'0 < {ObesityLevel} < 6')
    _Result = stats.spearmanr(_Data[FAF],_Data[ObesityLevel],alternative='less')

    print(f'Spearman Correlation Coefficient :: {_Result.statistic}\nP Value :: {_Result.pvalue}')
    return


@app.cell
def _(mo):
    mo.md(r"## 2.4. High Caloric Food")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        The consumption of high calorie foods is associated with obesity because they are non nutritious or ultra-processed foods. Therefore, the more they are consumed, the greater the likelihood of developing obesity or health problems. This can be shown by considering the trend of consumption, where as the complexity of obesity advances, the likelihood or consumption of high-calorie foods increases.
    
        When determining the coefficient of correspondence or association ($D = 0.4458$), it is shown that the association is significant and notable, therefore there is statistical evidence to indicate that the consumption of high calorie foods increases with the increase in the level of obesity. In other words, people with a higher obesity index tend to consume high-calorie foods frequently.
        """
    )
    return


@app.cell
def _(FAVC, ObesityDataset_1, src):
    PivotFAVC = src.SummaryCategoricalFeature(
        ObesityDataset_1,
        FAVC,
    )

    PivotFAVC
    return (PivotFAVC,)


@app.cell
def _(PivotFAVC, src):
    src.PlotPivotTable(
        PivotFAVC,
        'High Caloric Food',
    )
    return


@app.cell
def _(FAVC, ObesityDataset_1, ObesityLevel, stats):
    _Result = stats.somersd(ObesityDataset_1[FAVC],ObesityDataset_1[ObesityLevel])

    print(f'Somers Correspondence Coefficient :: {_Result.statistic}\nP Value :: {_Result.pvalue}')
    return


@app.cell
def _(mo):
    mo.md(r"# 3. Factor Analysis")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        When considering the first three factors, it is found that they allow to explain with greater understanding some of the attributes considered in [Data Analysis](#2-data-analysis). The following results show that the indicated attributes have a loading score greater than $0.5$ in absolute value:
    
        * Factor 1 [`Height`, `Weight`, `family_history_with_overweight`, `BMI`] focus on attributes related to the individual's physical condition and which are the main factors determining a person's level of obesity by considering that all relevant attributes have a positive loading on the factor. The latter can be contrasted by considering the [BMI analysis](#21body-mass-index-bmi).
    
        * Factor 2 [`Age`, `MTRANS_Public_Transportation`, `MTRANS_Automobile`] concerns the means of transportation aspect, where the use of public transportation and automobiles are in opposite directions, because the use of one determines, to some extent, the economic position of a person and his or her access to certain means of transportation. This factor is not relevant to the discussion concerning the level of obesity.
    
        * Factor 3 [`Gender`, `Height`] represents the phenomenon that men tend to be taller than women. Although height does play a relevant role in determining the level of obesity, it is not the only thing, so this factor remains more of a confirmation or statistical proof of a known fact in medicine.
    
        Of the factors studied, the first one concentrated most of the relevant information for the study, because its attributes allow explaining why a person is at a certain level of obesity, not only considering his/her weight and height but also his/her environment (close family with overweight).
        """
    )
    return


@app.cell
def _(Gender, MTRANS, ObesityDataset_1, ObesityLevel, src):
    # Applying auxiliar encodings to features

    MTRANS_Encode = list(map(str,'MTRANS_'+src.TransportationMethods))
    ObesityDataset_2 = ObesityDataset_1.copy(deep=True)

    ObesityDataset_2[Gender] = ObesityDataset_2[Gender].map(src.EncodeGenderValue)
    ObesityDataset_2[[*MTRANS_Encode]] = [*ObesityDataset_2[MTRANS].map(src.EncodeMTransValue)]
    ObesityDataset_2.drop(columns=[MTRANS,ObesityLevel],inplace=True)
    return (ObesityDataset_2,)


@app.cell
def _(Factor, ObesityDataset_2):
    # Calculating loadings and scores of FA

    Num_Factors = 3
    FactorAnalysis = Factor(ObesityDataset_2.to_numpy(),Num_Factors)
    FactorAnalysisResults = FactorAnalysis.fit()
    return FactorAnalysis, Num_Factors


@app.cell
def _(FactorAnalysis, Num_Factors, ObesityDataset_2, np):
    # Relevant features in each factor

    for _factor in range(Num_Factors):
        _filter_loadings = np.abs(FactorAnalysis.loadings[:,_factor])>0.5

        print(f'\nFactor {_factor+1} ::')
        for _feature , _load in zip(ObesityDataset_2.columns[_filter_loadings],FactorAnalysis.loadings[_filter_loadings,_factor]):
            print(f'{_feature} {_load:.4f}')
    return


@app.cell
def _(FactorAnalysis, ObesityDataset_2, src):
    src.PlotFactorAnalysisLoadings(FactorAnalysis.loadings,ObesityDataset_2.columns)
    return


@app.cell
def _(mo):
    mo.md(r"# 4. Cluster Analysis")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        In order to perform the cluster analysis, it was necessary to consider in the first instance that all the features must have the same range of values for an adequate clustering, for this reason `MinMaxScaler` was used, and, on the other hand, we have a high dimensionality dataset, therefore a `PCA` was performed to conserve the first $4$ principal components.
    
        K-Means was chosen as the clustering algorithm using $3$ clusters, it is done with this number of groups to capture the levels of obesity according to whether it is underweight or normal weight, overweight and obese.
    
        Since there is an overlap between clusters, due to a Silhouette coefficient of $0.3753$, clustering together with PCA did not allow an adequate separation between the expected groups. The latter could be related to the fact that the original `NObeyesdad` classes are not linearly separable and this can be compared by observing the plots generated by the principal components.
        """
    )
    return


@app.cell
def _(
    ColumnTransformer,
    KMeans,
    MinMaxScaler,
    NumericalFeatures,
    PCA,
    Pipeline,
    RANDOM_STATE,
):
    # Defining clustering with preprocessing pipeline

    PreprocessingFeatures = ColumnTransformer(
        [
            ('Normalization',MinMaxScaler(),NumericalFeatures)
        ],
        remainder='passthrough',
    )

    components = 4
    ClusteringAnalysis = Pipeline(
        [
            ('Preprocessing',PreprocessingFeatures),
            ('DimensionalReduction',PCA(n_components=components,random_state=RANDOM_STATE)),
            ('Clustering',KMeans(n_clusters=3,random_state=RANDOM_STATE)),
        ]
    )
    return (ClusteringAnalysis,)


@app.cell
def _(ClusteringAnalysis, ObesityDataset_2, silhouette_score):
    # Applying cluster analysis

    LabelsClusters = ClusteringAnalysis.fit_predict(ObesityDataset_2)
    TransformedDataset = ClusteringAnalysis[:2].transform(ObesityDataset_2)

    _score = silhouette_score(TransformedDataset,LabelsClusters)
    print(f'Silhouette Score :: {_score:.4f}')
    return LabelsClusters, TransformedDataset


@app.cell
def _(LabelsClusters, TransformedDataset, src):
    src.PlotClusterAnalysis(TransformedDataset,LabelsClusters)
    return


if __name__ == "__main__":
    app.run()
