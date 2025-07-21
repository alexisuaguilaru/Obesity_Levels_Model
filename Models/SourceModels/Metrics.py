from sklearn.metrics import f1_score

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