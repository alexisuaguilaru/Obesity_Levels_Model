from pickle import dump as ml_save , load as ml_load
from torch import save as nn_save , load as nn_load

from sklearn.pipeline import Pipeline
from torch import nn

def SaveModelML(
        Model: Pipeline,
        PathDir: str,
        NameModel: str,
    ) -> None:
    """
    Function for saving/dumping a 
    model/pipeline of scikit-learn
    into a dir with a given name.

    Parameters
    ----------
    Model: Pipeline
        Model/Pipeline to save/dump
    PathDir: str
        Path where is saved/dumped the model
    NameModel: str
        Name of the model
    """

    with open(f'{PathDir}{NameModel.replace(' ','')}.pkl','wb') as file_model:
        ml_save(Model,file_model,protocol=5)

def LoadModelML(
        PathDir: str,
        NameModel: str,
    ) -> Pipeline:
    """
    Function for loading a 
    model/pipeline of scikit-learn
    from a dir with a given name.

    Parameters
    ----------
    PathDir: str
        Path from where is loaded the model
    NameModel: str
        Name of the model to load

    Return
    ------
    Model: Pipeline
        Loaded model/pipeline
    """

    with open(f'{PathDir}{NameModel.replace(' ','')}.pkl','rb') as file_model:
        model = ml_load(file_model)
        return model

def SaveModelNN(
        Model: nn.Module,
        PathDir: str,
        NameModel: str,
    ) -> None:
    """
    Function for saving/dumping a 
    model/neural network of PyTorch
    into a dir with a given name.

    Parameters
    ----------
    Model: Pipeline
        Model/NN to save/dump
    PathDir: str
        Path where is saved/dumped the model
    NameModel: str
        Name of the model
    """

    nn_save(Model,f'{PathDir}{NameModel.replace(' ','')}.pt')

def LoadModelNN(
        PathDir: str,
        NameModel: str,
    ) -> nn.Module:
    """
    Function for loading a 
    model/neural network of PyTorch
    from a dir with a given name.

    Parameters
    ----------
    PathDir: str
        Path from where is loaded the model
    NameModel: str
        Name of the model to load

    Return
    ------
    Model: Pipeline
        Loaded model/NN
    """

    return nn_load(f'{PathDir}{NameModel.replace(' ','')}.pt',weights_only=False)