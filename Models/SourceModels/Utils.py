from math import sqrt
from os import cpu_count

import pandas as pd

def SplitFeatures(
        Dataset: pd.DataFrame,
    ) -> tuple[list[str],list[str],str]:
    """
    Function for splitting features into 
    numerical/float, categorical/integer 
    and target based on their datatypes

    Parameter
    ---------
    Dataset: pd.DataFrame
        Dataset which features are split based on datatypes

    Returns
    -------
    NumericalFeatures: list[str]
        Set of numerical/float features
    CategoricalFeatures: list[str]
        Set of categorical/integer features
    TargetFeature: str
        Target to predict
    """

    NumericalFeatures , CategoricalFeatures = [] , []
    for feature in Dataset.columns[:-1]:
        if Dataset[feature].dtype == 'int':
            CategoricalFeatures.append(feature)
        else:
            NumericalFeatures.append(feature)

    return NumericalFeatures , CategoricalFeatures , Dataset.columns[-1]

def GetNumJobs():
    """
    Function for getting the 
    number of available 
    threads/jobs 
    
    Return
    -----
    NumJobs: int
        Number of threads available
    """

    return int(sqrt(cpu_count()))