from .Utils import MapLevelObesity

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