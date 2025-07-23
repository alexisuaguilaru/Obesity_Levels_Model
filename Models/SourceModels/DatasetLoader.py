from torch.utils.data import Dataset
import pandas as pd

from torch import Tensor

class DatasetLoader(Dataset):
    def __init__(
            self,
            PathDataset: str,
            Features: list[str],
            Target: str,
            Device: str,
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
        Device: str
            Destination device of resulting fetching/getting instance
        """

        self.Dataset_pd = pd.read_csv(PathDataset)
        self.Features = Features
        self.Target = Target
        self.Device = Device

    def __len__(self) -> int:
        """
        Return
        ------
        SizeDataset: int 
            Size of the dataset
        """

        return len(self.Dataset_pd)

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

        Instance = self.Dataset_pd.iloc[Index]
        Instance_X = Tensor(Instance[self.Features])
        Label_y = Tensor([Instance[self.Target]])

        return Instance_X.to(self.Device) , Label_y.to(self.Device)