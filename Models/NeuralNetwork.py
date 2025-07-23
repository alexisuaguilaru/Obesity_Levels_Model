import marimo

__generated_with = "0.14.12"
app = marimo.App()


@app.cell
def _():
    # Importing auxiliar libraries

    import marimo as mo

    # Importing libraries

    from torch import nn , device , optim , no_grad , Tensor
    from torch.cuda import is_available

    import pandas as pd
    from torch.utils.data import Dataset , DataLoader

    # Importing Functions and Utils

    import SourceModels as src
    return (
        DataLoader,
        Dataset,
        Tensor,
        device,
        is_available,
        mo,
        nn,
        no_grad,
        optim,
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
def _(DataLoader, Dataset_Evaluation: "Dataset", Dataset_Train: "Dataset"):
    # Creating data loaders

    BatchSize = 16
    Dataloader_Train = DataLoader(Dataset_Train,batch_size=BatchSize,shuffle=True)
    Dataloader_Evaluation = DataLoader(Dataset_Evaluation,batch_size=BatchSize,shuffle=True)
    return BatchSize, Dataloader_Evaluation, Dataloader_Train


@app.cell
def _(mo):
    mo.md(r"# 2. Model Architecture")
    return


@app.cell
def _(mo):
    mo.md(r"A simple neural network of small size is defined to facilitate its training and reduce the amount of memory required. The current architecture consists of a hidden layer.")
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
                nn.Linear(25,10),
                nn.ReLU(),
                nn.Linear(10,7),
            )

        def forward(
                self,
                Instance_X: Tensor
            ) -> Tensor:

            Logits = self.NN(Instance_X)
            return Logits
    return (NeuralNetwork,)


@app.cell
def _(NeuralNetwork, TORCH_DEVICE):
    # Creating the model instance

    Model_NN = NeuralNetwork().to(TORCH_DEVICE)

    Model_NN
    return (Model_NN,)


@app.cell
def _(mo):
    mo.md(r"# 3. Model Training")
    return


@app.cell
def _(mo):
    mo.md(r"")
    return


@app.cell
def _():
    from torcheval.metrics import MulticlassF1Score
    from typing import Callable

    def F1_NN(
            NumClass: int = 7,
            Average: str = 'weighted',
        ) -> Callable:
        """
        """

        return MulticlassF1Score(num_classes=NumClass,average=Average)
    return (F1_NN,)


@app.function
def TrainLoop(
        DataLoader,
        Model,
        LossFunction,
        Optimizer,
        BatchSize,
        Device: str,
    ):
    """
    """

    Size = len(DataLoader.dataset)
    Model.train()

    for batch, data_train in enumerate(DataLoader):
        instance_X , label_y = data_train[0].to(Device) , data_train[1].to(Device)

        pred_labels = Model(instance_X)
        loss_data = LossFunction(pred_labels, label_y)

        loss_data.backward()
        Optimizer.step()
        Optimizer.zero_grad()

        if batch % 10 == 0:
            loss_value , current = loss_data.item() , batch*BatchSize+len(instance_X)
            print(f"Loss :: {loss_value:>7f}  [{current:>5d}/{Size:>5d}]")


@app.cell
def _(no_grad):
    def EvaluationLoop(
            DataLoader,
            Model,
            LossFunction,
            Metric,
            Device,
        ):
        """
        """

        Model.eval()
        NumBatches = len(DataLoader)
        TestLoss = 0

        with no_grad():
            for data_test in DataLoader:
                instance_X , label_y = data_test[0].to(Device) , data_test[1].to(Device)

                pred_labels = Model(instance_X)
                TestLoss += LossFunction(pred_labels,label_y).item()

                Metric.update(pred_labels.argmax(1),label_y)

        TestLoss /= NumBatches
        print(f"Test Error: \nF1: {(Metric.compute()*100):>0.1f}%, Avg loss: {TestLoss:>8f} \n")
        Metric.reset() 
    return (EvaluationLoop,)


@app.cell
def _(F1_NN, Model_NN, nn, optim):
    LearningRate = 1e-3
    Optimizer = optim.Adam(
            Model_NN.parameters(),
            lr=LearningRate,
        )

    LossFunction = nn.CrossEntropyLoss()

    Metric = F1_NN()
    return LossFunction, Metric, Optimizer


@app.cell
def _(
    BatchSize,
    Dataloader_Evaluation,
    Dataloader_Train,
    EvaluationLoop,
    LossFunction,
    Metric,
    Model_NN,
    Optimizer,
    TORCH_DEVICE,
):
    Epochs = 10
    for _t in range(Epochs):
        print(f' Epoch {_t+1} '.center(25,'-'))
        TrainLoop(Dataloader_Train,Model_NN,LossFunction,Optimizer,BatchSize,TORCH_DEVICE)
        EvaluationLoop(Dataloader_Evaluation,Model_NN,LossFunction,Metric,TORCH_DEVICE)
    return


if __name__ == "__main__":
    app.run()
