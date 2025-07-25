import marimo

__generated_with = "0.14.12"
app = marimo.App()


@app.cell
def _():
    # Importing auxiliar libraries

    import marimo as mo

    # Importing libraries

    import pandas as pd
    from torch import nn , device
    from torch.cuda import is_available
    import seaborn as sns
    import matplotlib.pyplot as plt

    # Importing Functions and Utils

    import SourceModels as src
    from sklearn.pipeline import Pipeline
    return Pipeline, device, is_available, mo, nn, pd, src


@app.cell
def _(device, is_available, src):
    # Defining useful variables

    PATH = './Models/'
    PATH_SAVE = PATH + 'SaveModels/'
    DatasetFilename = PATH + 'Dataset_{}.csv'

    NUM_JOBS = src.GetNumJobs()
    TORCH_DEVICE = device('cuda' if is_available() else 'cpu')
    return DatasetFilename, NUM_JOBS, PATH, PATH_SAVE, TORCH_DEVICE


@app.cell
def _(mo):
    mo.md(r"# Result Analysis and Model Comparison")
    return


@app.cell
def _(mo):
    mo.md(r"")
    return


@app.cell
def _(mo):
    mo.md(r"# 1. Load Datasets and Models")
    return


@app.cell
def _(mo):
    mo.md(r"The models trained and optimized in [Machine Learning](./MachineLearning.py) and [Neural Network](./NeuralNetwork.py) are loaded to be evaluated based on the evaluation and test datasets.")
    return


@app.cell
def _(PATH, pd):
    # Loading datasets

    _DatasetFilename = PATH + 'Dataset_{}.csv'

    Dataset_Evaluation: pd.DataFrame = None
    Dataset_Test: pd.DataFrame = None
    for _type_dataset in ['Evaluation','Test']:
        globals()[f'Dataset_{_type_dataset}'] = pd.read_csv(_DatasetFilename.format(_type_dataset),engine='pyarrow')
    return (Dataset_Evaluation,)


@app.cell
def _(Dataset_Evaluation: "pd.DataFrame", src):
    # Splitting features 

    NumericalFeatures , CategoricalFeatures , Target = src.SplitFeatures(Dataset_Evaluation)
    Features = [*NumericalFeatures,*CategoricalFeatures]
    return Features, Target


@app.cell
def _():
    # Defining model names and containers

    MLModelsName = [
        'Logistic Regression',
        'Random Forest',
        'SVM',
        'AdaBoost',
    ]

    NNModelName = 'NN'
    return MLModelsName, NNModelName


@app.cell
def _(MLModelsName, NUM_JOBS, PATH_SAVE, Pipeline, src):
    # Loading ML models

    MLModels: list[Pipeline] = []

    for _model_name in MLModelsName:
        _ml_model = src.LoadModelML(PATH_SAVE,_model_name)
        for _step_estimator in _ml_model:
            _step_estimator.n_jobs = NUM_JOBS*NUM_JOBS

        MLModels.append(_ml_model)
    return (MLModels,)


@app.cell
def _(NNModelName, PATH_SAVE, TORCH_DEVICE, nn, src):
    # Loading NN model

    from NeuralNetwork import NeuralNetwork

    NNModel: nn.Module = src.LoadModelNN(PATH_SAVE,NNModelName).to(TORCH_DEVICE)
    return (NNModel,)


@app.cell
def _(mo):
    mo.md(r"# 2. Evaluation Dataset")
    return


@app.cell
def _(mo):
    mo.md(r"")
    return


@app.cell
def _(
    Dataset_Evaluation: "pd.DataFrame",
    Features,
    MLModels: "list[Pipeline]",
    Target,
    src,
):
    _index = 3
    src.EvaluateMLModel(
        MLModels[_index],
        Dataset_Evaluation,
        Features,
        Target,
    ) , MLModels[_index]
    return


@app.cell
def _(
    DatasetFilename,
    Features,
    NNModel: "nn.Module",
    TORCH_DEVICE,
    Target,
    src,
):
    src.EvaluateNNModel(
        NNModel,
        DatasetFilename.format('Evaluation'),
        Features,
        Target,
        TORCH_DEVICE,
    )
    return


if __name__ == "__main__":
    app.run()
