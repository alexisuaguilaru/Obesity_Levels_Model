import pandas as pd

def SplitFeatures(
        Dataset: pd.DataFrame,
    ) -> tuple[list[str],list[str],str]:
    """
    Function for splitting features into 
    numerical, categorical and target based on 
    their datatypes

    Parameter
    ---------
    Dataset: pd.DataFrame
        Dataset which features are split based on datatypes

    Returns
    -------
    NumericalFeatures: list[str]
        Set of numerical features
    CategoricalFeatures: list[str]
        Set of categorical features
    TargetFeature: str
        Target to predict
    """

    NumericalFeatures , CategoricalFeatures = [] , []
    for feature in Dataset.columns[:-1]:
        if Dataset[feature].dtype == 'object':
            CategoricalFeatures.append(feature)
        else:
            NumericalFeatures.append(feature)

    return NumericalFeatures , CategoricalFeatures , Dataset.columns[-1]

def CapitalizeYesNoValues(
        Value: str,
    ) -> str:
    """
    Function for capitalizing `yes` 
    and `no`

    Parameter
    ---------
    Value: str
        String to evaluate

    Return
    ------
    ValueCapitalize: str
        Value correctly capitalized
    """

    first_letter = Value[0]
    if  (first_letter == 'y') or (first_letter == 'n'):
        return Value.capitalize()
    else:
        return Value