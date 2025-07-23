from torch.utils.data import Dataset
import pandas as pd

from torch import Tensor

class DatasetLoader(Dataset):
    def __init__(
            self,
            PathDataset: str,
            Features: list[str],
            Target: str,
        ):
        """
        Dataset Loader for fetching/getting 
        data from a csv file with PyTorch 
        using Pandas for loading the dataset
        
        Parameters
        ----------
        PathDataset: str
            Dataset source/file 
        Features: list[str]
            Features columns of a instance
        Target: str
            Target column of a instance
        """

        Dataset = pd.read_csv(PathDataset)
        self.Instances = Dataset[Features].to_numpy()
        self.Labels = Dataset[Target].to_numpy()

    def __len__(self) -> int:
        """
        Return
        ------
        SizeDataset: int 
            Size of the dataset
        """

        return len(self.Instances)

    def __getitem__(
            self,
            Index: int,
        ) -> tuple[Tensor,Tensor]:
        """
        Returns
        -------
        FeatureValues: Tensor
            Tensor with instance X
        Label: Tensor
            Tensor with instance label
        """

        Instance_X = Tensor(self.Instances[Index])
        Label_y = self.Labels[Index]

        return Instance_X , Label_y