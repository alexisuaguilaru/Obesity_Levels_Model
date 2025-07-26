import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from matplotlib.figure import Figure

def InitDataFrameResults() -> pd.DataFrame:
    """
    Function for creating a `pd.DataFrame` for 
    saving the results/scores of a model.

    Return
    ------
    DataFrameResults: pd.DataFrame
        Data frame for results
    """

    DataFrameResults = pd.DataFrame(columns=['Accuracy','Precision','Recall','F1'],dtype=float)
    DataFrameResults.rename_axis(index='Model',inplace=True)
    return DataFrameResults

def PlotResults(
        Results: pd.DataFrame,
        TypeDataset: str,    
    ) -> Figure:
    """
    Function for plot the results of 
    model evaluations.

    Parameters
    ----------
    Results: pd.DataFrame
        `pd.DataFrame` with the scores of each model
    TypeDataset: str
        Type of dataset used for getting the results

    Return
    ------
    Plot: Figure
        Plot with the results
    """

    Fig , Axes = plt.subplots(
        figsize=(9,7),
        subplot_kw={'frame_on':False,'ylim':(0.9,1)},
    )

    Results.plot(
        kind='bar',
        ax=Axes,
        legend=False,
        color=sns.color_palette('Set2'),
    )
    Axes.grid(True,axis='y',color='gray',lw=1,ls=':')

    Axes.set_yticks(np.linspace(0.9,1,11))
    TicksLabels = Axes.get_xticklabels()
    Axes.set_xticks(range(len(TicksLabels)),TicksLabels,rotation=30)

    Axes.tick_params(axis='both',labelsize=15,width=0)
    Axes.set_xlabel(Axes.get_xlabel(),size=17)
    Axes.set_ylabel('Score',size=17)
    Axes.set_title(f'Results on\n{TypeDataset} Dataset',size=24)

    Axes.legend(fontsize=14)

    return Fig