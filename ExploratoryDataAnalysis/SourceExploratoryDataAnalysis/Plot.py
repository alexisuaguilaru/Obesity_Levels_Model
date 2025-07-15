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

    fig , axes = plt.subplots(subplot_kw={'frame_on':False})

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

def PlotPivotTable(
        PivotTable: pd.DataFrame,
        Title: str,
        LabelsLegend: list[str] = ['No','Yes'],
    ) -> Figure:
    """
    Function for plotting the distribution 
    the results of a pivot table

    Parameter
    ---------
    PivotTable: pd.DataFrame
        Pivot table which is plotted
    Title: str
        Plot and legend title
    LabelsLegend: list[str]
        Label values for legend

    Return
    ------
    Plot : Figure
        Figure with the plot
    """

    fig , axes = plt.subplots(subplot_kw={'frame_on':False})
    PivotTable.plot(
        kind='bar',
        color=['#33D74B','#D10537'],
        ax=axes,
    )
    SetLabelAxisNames(axes,f'Distribution of {Title}',axes.get_xlabel(),'Count')
    axes.legend(title=Title,labels=LabelsLegend)

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

def PlotFactorAnalysisLoadings(
        Loadings: np.ndarray,
        Labels: list[str],
        Threshold: float = 0.5,
    ) -> Figure:
    """
    Function for generating the 
    loading factor plot using the 
    first two factors

    Parameters
    ----------
    Loadings: np.ndarray
        Loadings of each factor
    Labels: list[str]
        Feature names
    Threshold: float
        Threshold for relevant feature

    Return
    ------
    PlotLoadingFactor : Figure
        Figure of loading factor plot with relevant features 
    """

    fig , axes = plt.subplots(subplot_kw={'frame_on':False})
    relevant_features_index = np.any(np.abs(Loadings[:,:2]) > Threshold,axis=1)
    relevant_loadings = Loadings[relevant_features_index]

    sns.scatterplot(
        x=relevant_loadings[:,0],
        y=relevant_loadings[:,1],
        s=np.ones((relevant_loadings.shape[0],))*72,
        color=BASE_COLOR,
        alpha=0.75,
        legend=False,
        ax=axes,
    )
    for position , feature_name in zip(relevant_loadings,Labels[relevant_features_index]):
        axes.annotate(
            feature_name,
            position[:2],
            size=11,
        )

    SetLabelAxisNames(axes,'Loading Factor\nRelevant Features','Factor 1','Factor 2')

    return fig