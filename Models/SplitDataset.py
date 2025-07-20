import pandas as pd
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    # Read original dataset
    FullDataset = pd.read_csv(
        '../Datasets/ObesityDataset_Clean.csv',
        index_col=0,
    )

    RatioSplit = 0.8
    RandomState = 8013

    # Splitting into train and test datasets
    TrainDataset , TestDataset = train_test_split(
        FullDataset,
        train_size=RatioSplit,
        random_state=RandomState,
    )

    # Splitting into train and evaluation datasets
    TrainDataset , EvaluationDataset = train_test_split(
        TrainDataset,
        train_size=RatioSplit,
        random_state=RandomState,
    )

    # Saving datasets into CSV files
    for type_dataset in ['Train','Test','Evaluation']:
        dataset:pd.DataFrame = eval(f'{type_dataset}Dataset')
        dataset.to_csv(
            f'Dataset_{type_dataset}.csv',
            index=False,
        )