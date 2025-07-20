import optuna
from functools import partial
from copy import deepcopy

from sklearn.pipeline import Pipeline
import pandas as pd
from typing import Callable , Any

suggest_float = optuna.trial.Trial.suggest_float
suggest_int = optuna.trial.Trial.suggest_int
suggest_categorical = optuna.trial.Trial.suggest_categorical

class MachinLearningTrainer:
    def __init__(
            self,
            MLModel: Pipeline,
            ModelParams: dict[str,tuple],
            Metric: Callable,
        ):
        """
        Trainer for SciKit-Learn Estimators/Models 
        using Optuna. Inspired by `sklearn.model_selection.GridSearchCV` 
        interface/calling.

        Parameters
        ----------
        MLModel: Pipeline
            Model/Estimator to optimize its hyperparameters
        ModelParams: dict[str,tuple]
            Hyperparameters to optimize and their range of values. It is a dict where 
            the key is the parameter name and value is a tuple whose items are a data 
            type and a range of values
        Metric: Callable
            Metric for evaluating the model. It is a callable that accepts the model, 
            a dataset of instances and their labels
        """

        self.Model = MLModel
        self.Metric = Metric
        self.CreateGetSuggestsFunctions(ModelParams)

    def __call__(
            self,
            TrainDataset_X: pd.DataFrame,
            TrainDataset_y: pd.DataFrame,
            EvaluationDataset_X: pd.DataFrame,
            EvaluationDataset_y: pd.DataFrame,
            NumTrials: int = 10,
            NumJobs: int = 1,
        ):
        """
        Function for searching best parameters 
        for a model given a train dataset

        Parameters
        ----------
        TrainDataset_X: pd.DataFrame
            Instances of train dataset
        TrainDataset_y: pd.DataFrame
            True labels of train dataset
        EvaluationDataset_X: pd.DataFrame
            Instances of evaluation dataset
        EvaluationDataset_y: pd.DataFrame
            True labels of evaluation dataset
        NumTrials: int
            `n_trials` parameter for optuna.create_study
        NumJobs: int
            `n_jobs` parameter for optuna.create_study

        Returns
        -------
        BestParameters: dict[str,Any]
            Dict with best parameters founded by Optuna
        """

        Objective = self.GetObjectiveFunction(TrainDataset_X,TrainDataset_y,EvaluationDataset_X,EvaluationDataset_y)
        study = optuna.create_study(study_name='OptimizeModel',direction='maximize')
        study.optimize(Objective,n_trials=NumTrials,n_jobs=NumJobs)

        return study.best_params

    def CreateGetSuggestsFunctions(
            self,
            ModelParams: dict[str,tuple],
        ) -> None:
        """
        Function for defining functions 
        for suggest_* methods of `optuna.trial.Trial` 
        using dict of parameters in `__init__`.

        Parameter
        ---------
        ModelParams: dict[str,tuple]
            Hyperparameters to optimize defined by `__init__` parameter
        """

        SuggestsFunctions = {}
        for param_name , (type_suggest , params_suggest) in ModelParams.items():
            if type_suggest == 'float':
                param_low , param_high = params_suggest
                suggest_function = partial(suggest_float,name=param_name,low=param_low,high=param_high)

            elif type_suggest == 'int':
                param_low , param_high = params_suggest
                suggest_function = partial(suggest_int,name=param_name,low=param_low,high=param_high)

            elif type_suggest == 'categorical':
                suggest_function = partial(suggest_categorical,name=param_name,choices=params_suggest)

            else:
                raise Exception(f'{type_suggest} Not Implemented')

            SuggestsFunctions[param_name] = suggest_function

        self.SuggestsFunctions = SuggestsFunctions

    def GetObjectiveFunction(
            self,
            TrainDataset_X: pd.DataFrame,
            TrainDataset_y: pd.DataFrame,
            EvaluationDataset_X: pd.DataFrame,
            EvaluationDataset_y: pd.DataFrame,
        ) -> Callable:
        """
        Function for wrapping the definition of 
        objective function for Optuna.

        Parameters
        ----------
        TrainDataset_X: pd.DataFrame
            Instances of train dataset
        TrainDataset_y: pd.DataFrame
            True labels of train dataset
        EvaluationDataset_X: pd.DataFrame
            Instances of evaluation dataset
        EvaluationDataset_y: pd.DataFrame
            True labels of evaluation dataset

        Return
        ------
        ObjectiveFunction: Callable
            Objective function to optimize, 
            train and evaluate the base model
        """

        def ObjectiveFunction(
                Trial: optuna.trial.Trial,
            ) -> float:
            """
            Function for training the model using 
            the suggested parameters by Trial.

            Parameters
            ----------
            Trial: optuna.trial.Trial
                Trial from `optuna.study.Study` object

            Return
            ------
            ScoreMetric: float
                Metric function evaluates on the trained model and evaluation dataset
            """
            trainable_model = deepcopy(self.Model)
            suggested_parameters = self.GetSuggestedParameters(Trial)
            trainable_model.set_params(**suggested_parameters)

            trainable_model.fit(TrainDataset_X,TrainDataset_y)
            return self.Metric(trainable_model,EvaluationDataset_X,EvaluationDataset_y)

        return ObjectiveFunction

    def GetSuggestedParameters(
            self,
            Trial: optuna.trial.Trial,
        ) -> dict[str,Any]:
        """
        Function for getting a dict with the 
        suggested parameters by Trial.
        
        Parameter
        ---------
        Trial: optuna.trial.Trial
            Trial from `optuna.study.Study` object

        Return
        ------
        SuggestedParameters: dict[str,Any]
            Dict with the suggested parameters by Trial
        """

        SuggestedParameters = {}
        for param_name , suggest_function in self.SuggestsFunctions.items():
            SuggestedParameters[param_name] = suggest_function(Trial)

        return SuggestedParameters