import marimo

__generated_with = "0.14.12"
app = marimo.App()


@app.cell
def _():
    # Importing auxiliar libraries

    import marimo as mo

    # Importing libraries

    from functools import partial
    from torch import nn , device , optim , no_grad , Tensor
    from torch.cuda import is_available

    import pandas as pd
    from torch.utils.data import Dataset , DataLoader

    # Importing Functions and Utils

    import SourceModels as src
    return (
        Dataset,
        Tensor,
        device,
        is_available,
        mo,
        nn,
        optim,
        partial,
        pd,
        src,
    )


@app.cell
def _(device, is_available, src):
    # Defining useful variables

    PATH = './Models/'
    PATH_SAVE = PATH + 'SaveModels/'

    NUM_JOBS = src.GetNumJobs()
    TORCH_DEVICE = device('cuda' if is_available() else 'cpu')

    RANDOM_STATE = 8013
    return PATH, TORCH_DEVICE


@app.cell
def _(mo):
    mo.md(r"# Neural Network")
    return


@app.cell
def _(mo):
    mo.md(r"")
    return


@app.cell
def _(mo):
    mo.md(r"# 1. Load Dataset")
    return


@app.cell
def _(mo):
    mo.md(r"Subclassing of [`Dataset`](https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.Dataset) is used to load the dataset so that it can be compatible with the neural networks in PyTorch. After loading the datasets, the [`DataLoader`](https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) used to train and evaluate the models are created. When loading the dataset, no normalization or scaling transformation is applied to the dataset.")
    return


@app.cell
def _(PATH, pd, src):
    # Splitting features 

    DatasetFilename = PATH + 'Dataset_{}.csv'
    _Dataset = pd.read_csv(DatasetFilename.format('Train'),nrows=1)

    NumericalFeatures , CategoricalFeatures , Target = src.SplitFeatures(_Dataset)
    Features = [*NumericalFeatures,*CategoricalFeatures]

    del _Dataset
    return DatasetFilename, Features, Target


@app.cell
def _(Dataset, DatasetFilename, Features, Target, src):
    # Loading datasets

    Dataset_Train: Dataset = None
    Dataset_Evaluation: Dataset = None
    for _type_dataset in ['Train','Evaluation']:
        globals()[f'Dataset_{_type_dataset}'] = src.DatasetLoader(
            DatasetFilename.format(_type_dataset),
            Features,
            Target,
        )
    return Dataset_Evaluation, Dataset_Train


@app.cell
def _(mo):
    mo.md(r"# 2. Model Architecture")
    return


@app.cell
def _(mo):
    mo.md(r"A simple neural network of small size is defined to facilitate its training and reduce the amount of memory required.")
    return


@app.cell
def _(Tensor, nn):
    class NeuralNetwork(nn.Module):
        def __init__(self):
            """
            Neural network architecture for 
            predicting the obesity level of 
            a person
            """

            super().__init__()

            self.NN = nn.Sequential(
                nn.Linear(21,50),
                nn.ReLU(),
                nn.Linear(50,25),
                nn.ReLU(),
                nn.Linear(25,15),
                nn.ReLU(),
                nn.Linear(15,7),
            )

        def forward(
                self,
                Instance_X: Tensor
            ) -> Tensor:

            Logits = self.NN(Instance_X)
            return Logits
    return (NeuralNetwork,)


@app.cell
def _(NeuralNetwork):
    # Creating the model instance

    Model_NN = NeuralNetwork()

    Model_NN
    return


@app.cell
def _(mo):
    mo.md(r"# 3. Model Training")
    return


@app.cell
def _(mo):
    mo.md(r"")
    return


@app.cell
def _(nn, optim, partial, src):
    # Defining optimizer, loss function and metric

    _LearningRate = 1e-4
    Optimizer = partial(optim.Adam,lr=_LearningRate)

    LossFunction = nn.CrossEntropyLoss()
    MetricFunction = src.F1_NN()
    return LossFunction, MetricFunction, Optimizer


@app.cell
def _(LossFunction, NeuralNetwork, Optimizer, src):
    # Initializing trainer of Neural Network

    NN_Trainer = src.NeuralNetworTrainer(
            NeuralNetwork,
            Optimizer,
            LossFunction,
        )
    return (NN_Trainer,)


@app.cell
def _(
    Dataset_Evaluation: "Dataset",
    Dataset_Train: "Dataset",
    MetricFunction,
    NN_Trainer,
    TORCH_DEVICE,
):
    # Training of neural network

    _Epochs = 5
    BatchSize = 16
    NN_Trainer(
        Dataset_Train,
        Dataset_Evaluation,
        BatchSize,
        _Epochs,
        MetricFunction,
        TORCH_DEVICE
    )
    return


if __name__ == "__main__":
    app.run()
