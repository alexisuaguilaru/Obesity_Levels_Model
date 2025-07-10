import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from .Utils import *
from .Base import *

import pandas as pd
from matplotlib.figure import Figure
from matplotlib.axes import Axes

BASE_COLOR = '#501FA3'

def PlotFeatureOverCategories(
      Dataset: pd.DataFrame,
      Feature: str, 
      CategoryFeature: str|list[str] = 'NObeyesdad',
      ReorderCategories: list[str] = MapObesityLevel,
      FeatureName: str = None,
      CategoryName: str = None,
    ) -> Figure:
    """
    Function for plotting the distribution 
    of a numerical feature over a 
    categorical feature

    Parameters
    ----------
    Dataset: pd.DataFrame
        Dataset which is plotted
    Feature: str
        Numerical feature to plot
    CategoryFeature: str|list[str]
        Categorical feature to plot
    ReorderCategories: list[str]
        Map for reordering categories
    FeatureName: str
        Name label of numerical feature
    CategoryName: str
        Name label of categorical feature

    Return
    ------
    PlotDistribution : Figure
        Figure with the distribution plot 
    """

    fig , axes = plt.subplots()

    sns.boxplot(
        Dataset,
        x=Feature,
        y=CategoryFeature,
        order=ReorderCategories,
        color=BASE_COLOR,
        fill=False,
        linewidth=1,
        fliersize=5,
        ax=axes,
    )
    SetLabelAxisNames(axes,f'Distribution of {FeatureName}',FeatureName,CategoryName)

    return fig

def PlotRelativesWithOverweight(
    Dataset: pd.DataFrame,
    ) -> Figure:
    """
    Function for plotting the distribution 
    of `family_history_with_overweight` 
    feature over `NObeyesdad`

    Parameter
    ---------
    Dataset: pd.DataFrame
        Dataset which is plotted

    Return
    ------
    PlotDistribution : Figure
        Figure with the distribution plot 
    """

    fig , axes = plt.subplots()

    sns.countplot(
        Dataset,
        x=ObesityLevel,
        hue=FamilyOverweight,
        ax=axes,
        palette=['#33D74B','#D10537']
    )

    SetLabelAxisNames(axes,'Distribution of Relatives with Overweight','Obesity Level','Count')
    axes.set_xticks(np.arange(7),MapLevelObesity.values(),rotation=45)
    axes.legend(title='Relatives with\nOverweight',labels=['No','Yes'])

    return fig

def SetLabelAxisNames(
        Axes: Axes,
        Title: str = None,
        XLabel: str = None,
        YLabel: str = None,
    ) -> None:
    """
    Function for setting title and 
    label axis of a plot

    Parameters
    ----------
    Axes: Axes
        Axes where the plot is
    Title: str
        Plot title
    XLabel: str
        Plot X axis label
    YLabel: str
        Plot Y axis label
    """

    if Title: Axes.set_title(Title,size=16)
    if XLabel: Axes.set_xlabel(XLabel,size=12)
    if YLabel: Axes.set_ylabel(YLabel,size=12)

    Axes.tick_params(axis='both',labelsize=11)