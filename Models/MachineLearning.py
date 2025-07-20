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
    return ColumnTransformer, Pipeline, StandardScaler, mo, pd, src


@app.cell
def _(src):
    # Defining useful variables

    PATH = './Models/'
    NUM_JOBS = src.GetNumJobs()

    RANDOM_STATE = 8013
    return NUM_JOBS, PATH, RANDOM_STATE


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
    Dataset_Evaluation: pd.DataFrame = None
    for _type_dataset in ['Train','Evaluation']:
        globals()[f'Dataset_{_type_dataset}'] = pd.read_csv(_DatasetFilename.format(_type_dataset),engine='pyarrow')
    return Dataset_Evaluation, Dataset_Train


@app.cell
def _(Dataset_Train: "pd.DataFrame", src):
    # Splitting features 

    NumericalFeatures , CategoricalFeatures , Target = src.SplitFeatures(Dataset_Train)
    Features = [*NumericalFeatures,*CategoricalFeatures]
    return Features, NumericalFeatures, Target


@app.cell
def _(ColumnTransformer, NUM_JOBS, NumericalFeatures, StandardScaler):
    # Preprocessing pipeline

    PreprocessingPipeline = ColumnTransformer(
        [
            ('NumericalFeatures',StandardScaler(),NumericalFeatures),
        ],
        remainder='passthrough',
        n_jobs=NUM_JOBS,
    )

    PreprocessingPipeline
    return (PreprocessingPipeline,)


@app.cell
def _(mo):
    mo.md(r"# 2. Models Definition")
    return


@app.cell
def _(mo):
    mo.md(r"")
    return


@app.cell
def _(mo):
    mo.md(r"## 2.1. Logistic Regression")
    return


@app.cell
def _(mo):
    mo.md(r"In order to generate more flexibility in the hyperparameter fine-tuning, it was decided to use the penalty [`elasticnet`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) that allows using a convex combination of the `l1` and `l2` penalties, so that the optimizer can choose and give a higher weight to the most convenient penalty for the classification problem.")
    return


@app.cell
def _(NUM_JOBS, Pipeline, PreprocessingPipeline, RANDOM_STATE):
    # Defining Logistic Regression model

    from sklearn.linear_model import LogisticRegression

    LogisticRegression_Model = Pipeline(
        [
            ('Preprocessing',PreprocessingPipeline),
            ('Model',LogisticRegression(
                penalty='elasticnet',
                solver='saga',
                random_state=RANDOM_STATE,
                n_jobs=NUM_JOBS,
                )
            ),
        ]
    )

    LogisticRegression_Parameters = {
        'Model__C':('float',[1e-10,2]),
        'Model__l1_ratio':('float',[0,1]),
    }


    LogisticRegression_Model
    return


@app.cell
def _(mo):
    mo.md(r"## 2.2. Random Forest")
    return


@app.cell
def _(mo):
    mo.md(r"Fine-tunning is performed on the most relevant hyperparemeters of [Random Fores](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) which are: `n_estimators`,`max_depth` and `criterion`. They are relevant since they allow to control the overfit and underfit of the model.")
    return


@app.cell
def _(NUM_JOBS, Pipeline, PreprocessingPipeline, RANDOM_STATE):
    # Defining Random Forest model

    from sklearn.ensemble import RandomForestClassifier

    RandomForest_Model = Pipeline(
        [
            ('Preprocessing',PreprocessingPipeline),
            ('Model',RandomForestClassifier(
                random_state=RANDOM_STATE,
                n_jobs=NUM_JOBS,
                )
            ),
        ]
    )

    RandomForest_Parameters = {
        'Model__n_estimators': ('int',[1,100]),
        'Model__max_depth': ('int',[1,12]),
        'Model__criterion': ('categorical',['gini','entropy'])
    }


    RandomForest_Model
    return RandomForest_Model, RandomForest_Parameters


@app.cell
def _(mo):
    mo.md(r"# 3. Models Fitting")
    return


@app.cell
def _(
    Dataset_Evaluation: "pd.DataFrame",
    Dataset_Train: "pd.DataFrame",
    Features,
    NUM_JOBS,
    Pipeline,
    RandomForest_Model,
    RandomForest_Parameters,
    Target,
    pd,
    src,
):
    from sklearn.metrics import f1_score

    def Metric(
            Model: Pipeline,
            Dataset_X: pd.DataFrame,
            Dataset_y: pd.DataFrame,
        ):
        """
        """
        PredictLabels = Model.predict(Dataset_X)
        return f1_score(Dataset_y,PredictLabels,average='weighted')


    _test = src.MachinLearningTrainer(
        RandomForest_Model,
        RandomForest_Parameters,
        Metric,
    )

    import warnings
    from sklearn.exceptions import ConvergenceWarning

    with warnings.catch_warnings():
        warnings.simplefilter('ignore',category=ConvergenceWarning)
        best_params = _test(
            Dataset_Train[Features],
            Dataset_Train[Target],
            Dataset_Evaluation[Features],
            Dataset_Evaluation[Target],
            NumTrials=32,
            NumJobs=NUM_JOBS,
        )
    return


if __name__ == "__main__":
    app.run()
