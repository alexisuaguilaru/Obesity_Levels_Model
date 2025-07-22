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
    PATH_SAVE = PATH + 'SaveModels/'

    NUM_JOBS = src.GetNumJobs()

    RANDOM_STATE = 8013
    return NUM_JOBS, PATH, RANDOM_STATE


@app.cell
def _(mo):
    mo.md(r"# Machine Learning Models")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        The detailed procedure to generate Machine Learning models for the classification of a patient's obesity level based on their lifestyle habits is presented in this notebook. For the fine-tunning of the hyperparameters, [Optuna](https://optuna.org) is used together with the training dataset and the models are evaluated on the evaluation dataset.
    
        Section [1. Preprocessing Pipeline](#1-preprocessing-pipeline) defines the preprocessing of the real value features, which are applied a standard scaling, and the other features are left unchanged. This scaling is done so that all the features are in the same range and to avoid bias problems that can be generated.
    
        In section [2. Models Definition](#2-models-definition) the models to be used are created, where the priority is to have a diversification of classification techniques, together with the space of hyperparameters to be fine-tuned by means of [Optuna](https://optuna.org). In addition, a brief justification of the choice of the hyperparameters to be optimized is presented based on the training flexibility of the models (so that they fit adequately to the training set).
    
        Finally, in section [3. Models Fitting](#3-models-fitting) the hyperparameters of each model are fitted and the models are trained with the best hyperparameters found. For the evaluation of the models, F1 score was used due to the imbalance in the dataset with respect to `NObeyesdad`.
        """
    )
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
    mo.md(
        r"""
        In this section the candidate models to be trained are defined, where both linear and nonlinear models are used to generate greater flexibility when solving the classification problem. The hyperparameters to be optimized during training and fine-tunning are also defined.
    
        The models that were chosen represent a reduced collection of techniques and ways of approaching the classification problem, where the priority was to have a greater diversification of them. Specifically, the following were chosen:
    
        * [Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
        * [Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
        * [SVM](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)
        * [AdaBoost](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html)
        """
    )
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
    return LogisticRegression_Model, LogisticRegression_Parameters


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
    mo.md(r"## 2.3. Support Vector Machine (SVM)")
    return


@app.cell
def _(mo):
    mo.md(r"The most important hyperparameter in [SVM](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html) is the `kernel`, because it controls the nonlinearity of the algorithm; and for more flexibility during training other parameters are considered for the kernel of the model.")
    return


@app.cell
def _(Pipeline, PreprocessingPipeline, RANDOM_STATE):
    # Defining Random Forest model

    from sklearn.svm import SVC

    SVM_Model = Pipeline(
        [
            ('Preprocessing',PreprocessingPipeline),
            ('Model',SVC(
                random_state=RANDOM_STATE,
                )
            ),
        ]
    )

    SVM_Parameters = {
        'Model__C':('float',[1e-10,2]),
        'Model__kernel':('categorical',['poly','rbf','sigmoid']),
        'Model__degree':('int',[1,5]),
        'Model__gamma':('float',[1e-10,2]),
        'Model__coef0':('float',[0,2]),
    }


    SVM_Model
    return SVM_Model, SVM_Parameters


@app.cell
def _(mo):
    mo.md(r"## 2.4. Adaptive Boosting (AdaBoost)")
    return


@app.cell
def _(mo):
    mo.md(r"AdaBoost is used as an ensemble model where the number of estimators (`n_estimators`) is the main hyperparameter to optimize, being which allows to control the general underfit and overfit of the model.")
    return


@app.cell
def _(Pipeline, PreprocessingPipeline, RANDOM_STATE):
    # Defining Random Forest model

    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.tree import DecisionTreeClassifier

    AdaBoost_Model = Pipeline(
        [
            ('Preprocessing',PreprocessingPipeline),
            ('Model',AdaBoostClassifier(
                random_state=RANDOM_STATE,
                )
            ),
        ]
    )

    base_estimators = [DecisionTreeClassifier(max_depth=depth,random_state=RANDOM_STATE) for depth in range(1,3)]
    AdaBoost_Parameters = {
        'Model__estimator':('categorical',base_estimators),
        'Model__n_estimators':('int',[1,100]),
        'Model__learning_rate':('float',[1e-12,2]),
    }


    AdaBoost_Model
    return AdaBoost_Model, AdaBoost_Parameters


@app.cell
def _(mo):
    mo.md(r"# 3. Models Fitting")
    return


@app.cell
def _(mo):
    mo.md(r"With the definition of the models and hyperparameters to be optimized, the model fitting is performed using [Optuna](https://optuna.org) as framework to search for the best hyperparameters of each model according to the search space defined in [2. Models Definition](#2-models-definition). For determining the best hyperparameters, F1 score with weighted average is used because the dataset is slightly imbalanced with respect to the target (`NObeyesdad`), as described in [Exploratory Data Analysis](../ExploratoryDataAnalysis/ExploratoryDataAnalysis.py). Finally, the models are trained with the best hyperparameters and then saved.")
    return


@app.cell
def _(
    AdaBoost_Model,
    AdaBoost_Parameters,
    LogisticRegression_Model,
    LogisticRegression_Parameters,
    RandomForest_Model,
    RandomForest_Parameters,
    SVM_Model,
    SVM_Parameters,
):
    # Defining containers for models and their params to optimize, 
    # and variables for saving best models

    ModelsName = [
        'Logistic Regression',
        'Random Forest',
        'SVM',
        'AdaBoost',
    ]

    ModelsParams = [
        (LogisticRegression_Model , LogisticRegression_Parameters),
        (RandomForest_Model , RandomForest_Parameters),
        (SVM_Model , SVM_Parameters),
        (AdaBoost_Model , AdaBoost_Parameters)
    ]

    BestModels = []
    return BestModels, ModelsName, ModelsParams


@app.cell
def _(
    BestModels,
    Dataset_Evaluation: "pd.DataFrame",
    Dataset_Train: "pd.DataFrame",
    Features,
    ModelsName,
    ModelsParams,
    NUM_JOBS,
    Target,
    src,
):
    # Importing auxiliars para ignore warnings

    import warnings
    from sklearn.exceptions import ConvergenceWarning

    # Fine-tunning and training of models

    from copy import deepcopy

    _NumTrials = 8
    _Metric = src.F1_ML
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',category=ConvergenceWarning)
        warnings.simplefilter('ignore',category=UserWarning)

        TrainDataset_X = Dataset_Train[Features]
        TrainDataset_y = Dataset_Train[Target]
        EvaluationDataset_X = Dataset_Evaluation[Features]
        EvaluationDataset_y = Dataset_Evaluation[Target]

        for (_model , _params) , _model_name in zip(ModelsParams,ModelsName):
            # Defining optimizer
            _trainer = src.MachinLearningTrainer(
                _model,
                _params,
                _Metric,
            )

            # Fine-tuning of hyperparameters
            print(f' Start Fine-Tuning of {_model_name} '.center(50,'='))
            _best_params = _trainer(
                TrainDataset_X,
                TrainDataset_y,
                EvaluationDataset_X,
                EvaluationDataset_y,
                NumTrials=_NumTrials,
                NumJobs=NUM_JOBS,
            )

            # Training model with the best parameters
            _best_model = deepcopy(_model)
            _best_model.set_params(**_best_params)
            _best_model.fit(TrainDataset_X,TrainDataset_y)
            BestModels.append(deepcopy(_best_model))

        print('\n',' Start Models Evaluation '.center(50,'='))
        for _best_model , _model_name in zip(BestModels,ModelsName):
            _score = _Metric(_best_model,EvaluationDataset_X,EvaluationDataset_y)
            print(f'Best {_model_name} Model obtains :: {_score} Score')
    return


@app.cell
def _(BestModels, ModelsName):
    # Saving models

    for _best_model , _model_name in zip(BestModels,ModelsName):
        print('\n',f' Start Save {_model_name} Model '.center(50,'='))
        # src.SaveModelML(_best_model,PATH_SAVE,_model_name.replace(' ',''))
    return


if __name__ == "__main__":
    app.run()
