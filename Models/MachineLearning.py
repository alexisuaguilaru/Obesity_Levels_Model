import marimo

__generated_with = "0.14.12"
app = marimo.App()


@app.cell
def _():
    # Importing auxiliar libraries

    import marimo as mo

    # Importing libraries

    import pandas as pd
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler

    # Importing Functions and Utils

    import SourceModels as src
    return ColumnTransformer, StandardScaler, mo, pd, src


@app.cell
def _(src):
    # Defining useful variables

    PATH = './Models/'
    NUM_JOBS = src.GetNumJobs()
    return NUM_JOBS, PATH


@app.cell
def _(mo):
    mo.md(r"# Machine Learning Models")
    return


@app.cell
def _(mo):
    mo.md(r"")
    return


@app.cell
def _(mo):
    mo.md(r"# 1. Preprocesising Pipeline")
    return


@app.cell
def _(mo):
    mo.md(r"After the treatment of the dataset in the EDA, only a standard scaling (standardization) is applied to the numerical features so that most of the features are in a similar range, this will favor both the training and the predictions of the models due to the reduction of the bias contained in the features.")
    return


@app.cell
def _(PATH, pd):
    # Loading datasets

    _DatasetFilename = PATH + 'Dataset_{}.csv'

    Dataset_Train: pd.DataFrame = None
    Dataset_Test: pd.DataFrame = None
    for _type_dataset in ['Train','Test']:
        globals()[f'Dataset_{_type_dataset}'] = pd.read_csv(_DatasetFilename.format(_type_dataset),engine='pyarrow')
    return (Dataset_Train,)


@app.cell
def _(Dataset_Train: "pd.DataFrame", src):
    # Splitting features 

    NumericalFeatures , CategoricalFeatures , Target = src.SplitFeatures(Dataset_Train)
    return (NumericalFeatures,)


@app.cell
def _(ColumnTransformer, NUM_JOBS, NumericalFeatures, StandardScaler):
    # Preprocessing pipeline

    PreprocessingPipeline = ColumnTransformer(
        [
            ('NumericalFeatures',StandardScaler(),NumericalFeatures),
        ],
        remainder='passthrough',
        n_jobs=NUM_JOBS**2,
    )

    PreprocessingPipeline
    return


if __name__ == "__main__":
    app.run()
