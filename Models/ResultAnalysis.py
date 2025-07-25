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

    NUM_JOBS = src.GetNumJobs()
    TORCH_DEVICE = device('cuda' if is_available() else 'cpu')
    return NUM_JOBS, PATH, PATH_SAVE, TORCH_DEVICE


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
    return


@app.cell
def _(Pipeline):
    # Defining model names and containers

    MLModelsName = [
        'Logistic Regression',
        'Random Forest',
        'SVM',
        'AdaBoost',
    ]
    MLModels: list[Pipeline] = []

    NNModelName = 'NN'
    return MLModels, MLModelsName, NNModelName


@app.cell
def _(MLModels: "list[Pipeline]", MLModelsName, NUM_JOBS, PATH_SAVE, src):
    # Loading ML models

    for _model_name in MLModelsName:
        _ml_model = src.LoadModelML(PATH_SAVE,_model_name)
        for _step_estimator in _ml_model:
            _step_estimator.n_jobs = NUM_JOBS*NUM_JOBS

        MLModels.append(_ml_model)
    return


@app.cell
def _(NNModelName, PATH_SAVE, TORCH_DEVICE, nn, src):
    # Loading NN model

    from NeuralNetwork import NeuralNetwork

    NNModel: nn.Module = src.LoadModelNN(PATH_SAVE,NNModelName).to(TORCH_DEVICE)
    return


if __name__ == "__main__":
    app.run()
