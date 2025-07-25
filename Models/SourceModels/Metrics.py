from sklearn.metrics import f1_score
from torcheval.metrics import MulticlassF1Score

from typing import Callable
from sklearn.pipeline import Pipeline
import pandas as pd

def F1_ML(
        Model: Pipeline,
        Dataset_X: pd.DataFrame,
        Dataset_y: pd.DataFrame,
    ):
    """
    Function for calculating F1 score of a 
    scikit-learn model/pipeline based on a dataset.

    Parameters
    ----------
    Model: Pipeline
        Model to evaluate
    Dataset_X: pd.DataFrame
        Instances of dataset to predict their labels
    Dataset_y: pd.DataFrame
        True labels of instances

    Return
    ------
    F1_score: float
        F1 score obtained by the model
    """

    PredictLabels = Model.predict(Dataset_X)
    return f1_score(Dataset_y,PredictLabels,average='weighted')

def F1_NN(
        NumClass: int = 7,
        Average: str = 'weighted',
    ) -> Callable:
    """
    Function for calculating F1 score of a 
    PyTorch model/nn.Module.

    Parameters
    ----------
    NumClass: int
        `num_clases` parameter of `torcheval.metrics.MulticlassF1Score`
    Average: str
        `average` parameter of `torcheval.metrics.MulticlassF1Score`

    Return
    ------
    F1Metric: Callable
        A object (instance) of `torcheval.metrics.MulticlassF1Score`
    """

    return MulticlassF1Score(num_classes=NumClass,average=Average)