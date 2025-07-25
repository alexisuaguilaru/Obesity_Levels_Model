from sklearn.metrics import accuracy_score , precision_recall_fscore_support
from torcheval.metrics import MulticlassAccuracy , MulticlassPrecision , MulticlassRecall , MulticlassF1Score
from torch.utils.data import DataLoader
from .DatasetLoader import DatasetLoader

from sklearn.pipeline import Pipeline
import pandas as pd
from torch import nn

def EvaluateMLModel(
        Model: Pipeline,
        Dataset: pd.DataFrame,
        Features: list[str],
        Target: str,
    ) -> list[float]:
    """
    Function for calculating accuracy, 
    precision, recall and f1 scores of 
    a scikit-learn model/pipeline 
    based on a dataset.

    Parameters
    ----------
    Model: Pipeline
        Model to evaluate
    Dataset: pd.DataFrame
        Dataset used to evaluate
    Features: list[str]
        Features columns of a instance
    Target: str
        Target column of a instance

    Return
    ------
    MetricScores: list[float]
        List of scores obtained by the model
    """

    Instances_X = Dataset[Features]
    Labels_y = Dataset[Target]
    Pred_y = Model.predict(Instances_X)

    *metrics , _ = precision_recall_fscore_support(Labels_y,Pred_y,average='weighted')
    return accuracy_score(Labels_y,Pred_y) , *metrics

def EvaluateNNModel(
        Model: nn.Module,
        DatasetPath: str,
        Features: list[str],
        Target: str,
        Device: str,
        BatchSize: int = 32,
        NumClass: int = 7,
        Average: str = 'weighted',
    ) -> list[float]:
    """
    Function for calculating accuracy, 
    precision, recall and f1 scores of 
    a PyTorch model/nn.Module. 
    based on a dataset.

    Parameters
    ----------
    Model: nn.Module
        Model to evaluate
    DatasetPath: str
        Dataset source/file 
    Features: list[str]
        Features columns of a instance
    Target: str
        Target column of a instance
    Device: str
        Location/Destination for instances and labels
    BatchSize: int
        `batch_siz` parameter of `DataLoader`
    NumClass: int
        `num_clases` parameter of `torcheval.metrics.Multiclass*`
    Average: str
        `average` parameter of `torcheval.metrics.Multiclass*`. Type of average
    """

    Accuracy = MulticlassAccuracy(num_classes=NumClass,average='micro')
    Precision = MulticlassPrecision(num_classes=NumClass,average=Average)
    Recall = MulticlassRecall(num_classes=NumClass,average=Average)
    F1Score = MulticlassF1Score(num_classes=NumClass,average=Average)
    Metrics = [Accuracy,Precision,Recall,F1Score,]

    Dataset = DatasetLoader(DatasetPath,Features,Target)
    for data in DataLoader(Dataset,batch_size=BatchSize):
        instance_X , label_y = data[0].to(Device) , data[1].to(Device)
        pred_labels = Model(instance_X)

        for metric in Metrics:
            metric.update(pred_labels.argmax(1),label_y)

    return [float(metric.compute()) for metric in Metrics]