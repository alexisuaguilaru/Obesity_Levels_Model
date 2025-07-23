import marimo

__generated_with = "0.14.12"
app = marimo.App()


@app.cell
def _():
    # Importing auxiliar libraries

    import marimo as mo

    # Importing libraries

    from torch import nn , device , Tensor
    from torch.cuda import is_available

    import pandas as pd
    from torch.utils.data import Dataset , DataLoader

    # Importing Functions and Utils

    import SourceModels as src
    return DataLoader, Dataset, device, is_available, mo, pd, src


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
def _(Dataset, DatasetFilename, Features, TORCH_DEVICE, Target, src):
    # Loading datasets

    Dataset_Train: Dataset = None
    Dataset_Evaluation: Dataset = None
    for _type_dataset in ['Train','Evaluation']:
        globals()[f'Dataset_{_type_dataset}'] = src.DatasetLoader(
            DatasetFilename.format(_type_dataset),
            Features,
            Target,
            TORCH_DEVICE
        )
    return Dataset_Evaluation, Dataset_Train


@app.cell
def _(DataLoader, Dataset_Evaluation: "Dataset", Dataset_Train: "Dataset"):
    # Creating data loaders

    BatchSize = 32
    Dataloader_Train = DataLoader(Dataset_Train,batch_size=BatchSize,shuffle=True)
    Dataloader_Evaluation = DataLoader(Dataset_Evaluation,batch_size=BatchSize,shuffle=True)
    return


if __name__ == "__main__":
    app.run()
