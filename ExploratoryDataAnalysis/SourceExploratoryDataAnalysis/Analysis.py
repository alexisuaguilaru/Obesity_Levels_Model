from .Utils import *

import pandas as pd

def SummaryStatisticsFeature(
        Dataset: pd.DataFrame,
        SummaryFeature: str,
        CategoryFeature: str = 'NObeyesdad',
        RenameCategory: dict[int|str,str] = MapLevelObesity,
        NameCategory: str = 'Obesity Level',
    ) -> pd.DataFrame:
    """
    Function for summarizing a numerical 
    feature based on categories

    Parameters
    ----------
    Dataset: pd.DataFrame
        Dataset which data is summarized
    SummaryFeature: str
        Feature to summary
    CategoryFeature: str
        Feature used for categories
    RenameCategory: dict[int|str,str]
        Map for renaming category names
    NameCategory: str
        Name label for `CategoryFeature`

    Return
    ------
    Summary: pd.DataFrame
        DataFrame with the summarized data
    """

    Summary = Dataset.groupby(CategoryFeature)[SummaryFeature].describe()[['mean','std','25%','50%','75%']]
    Summary.rename(index=RenameCategory,inplace=True)
    Summary.rename_axis(index=NameCategory,inplace=True)

    return Summary

def SummaryCategoricalFeature(
        Dataset: pd.DataFrame,
        SummaryFeatureColumns: str,
        SummaryFeatureRows: str = 'NObeyesdad',
        AuxiliarFeature: str = 'Age',
        RenameColumns: dict[int|str,str] = MapValueBinary,
        RenameRows: dict[int|str,str] = MapLevelObesity,
        NameLabelRows: str = 'Obesity Level',
    ) -> pd.DataFrame:
    """
    Function for summarizing a categorical
    feature based on categories

    Parameters
    ----------
    Dataset: pd.DataFrame
        Dataset which data is summarized
    SummaryFeatureColumns: str
        Feature to summary
    SummaryFeatureRows: str
        Feature used for categories
    RenameColumns: dict[int|str,str]
        Map for renaming axis of columns categories
    RenameRows: dict[int|str,str]
        Map for renaming axis of rows categories
    NameLabelRows: str
        Name label for `SummaryFeatureRows`

    Return
    ------
    Summary: pd.DataFrame
        DataFrame with the summarized data
    """

    Summary = Dataset.pivot_table(
        values=AuxiliarFeature,
        index=SummaryFeatureRows,
        columns=SummaryFeatureColumns,
        aggfunc='count',
        fill_value=0,
    )
    
    Summary.rename(
        columns=RenameColumns,
        index=RenameRows,
        inplace=True,
    )
    
    Summary.rename_axis(
        index=NameLabelRows,
        inplace=True,
    )

    return Summary