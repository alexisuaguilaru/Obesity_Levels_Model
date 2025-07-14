import pandas as pd
import numpy as np

def SplitFeatures(
        Dataset: pd.DataFrame,
    ) -> tuple[list[str],list[str],str]:
    """
    Function for splitting features into 
    numerical, categorical and target based on 
    their datatypes

    Parameter
    ---------
    Dataset: pd.DataFrame
        Dataset which features are split based on datatypes

    Returns
    -------
    NumericalFeatures: list[str]
        Set of numerical features
    CategoricalFeatures: list[str]
        Set of categorical features
    TargetFeature: str
        Target to predict
    """

    NumericalFeatures , CategoricalFeatures = [] , []
    for feature in Dataset.columns[:-1]:
        if Dataset[feature].dtype == 'object':
            CategoricalFeatures.append(feature)
        else:
            NumericalFeatures.append(feature)

    return NumericalFeatures , CategoricalFeatures , Dataset.columns[-1]

def CapitalizeYesNoValues(
        Value: str,
    ) -> str:
    """
    Function for capitalizing `yes` 
    and `no`

    Parameter
    ---------
    Value: str
        String to evaluate

    Return
    ------
    ValueCapitalize: str
        Value correctly capitalized
    """

    first_letter = Value[0]
    if  (first_letter == 'y') or (first_letter == 'n'):
        return Value.capitalize()
    else:
        return Value

MapValueBinary = {value:binary for value , binary in enumerate(['No','Yes'])}
def EncodeBinaryValue(
        BinValue: str,
    ) -> int:
    """
    Function for encoding 
    binary (`no`, `yes`) values

    Parameter
    ---------
    BinValue: str
        Binary value to encode

    Return
    ------
    Encoded_BinValue: int
        Encoded binary value
    """

    return int(BinValue[0] == 'Y')

Frequencies = ['No','Sometimes','Frequently','Always']
MapFrequencyValue = {frequency:value for value , frequency in enumerate(Frequencies)}
def EncodeFrequencyValue(
        FreqValue: str,
    ) -> int:
    """
    Function for encoding 
    frequencies 

    Parameter
    ---------
    FreqValue: str
        Frequency value

    Return
    ------
    Encoded_FreqValue: int
        Encoded frequency value
    """

    return MapFrequencyValue[FreqValue]

ObesityLevels = [
    'Insufficient_Weight','Normal_Weight',
    'Overweight_Level_I','Overweight_Level_II',
    'Obesity_Type_I','Obesity_Type_II','Obesity_Type_III'
]
MapObesityLevel = {obesity:level for level , obesity in enumerate(ObesityLevels)}
MapLevelObesity = {level:obesity for level , obesity in enumerate(ObesityLevels)}
def EncodeObesityLevel(
        ObesityLevel: str,
    ) -> int:
    """
    Function for encoding 
    obesity levels

    Parameter
    ---------
    ObesityLevel: str
        Obesity level to encode

    Return
    ------
    Encoded_ObesityLevel: int
        Encoded obesity level
    """

    return MapObesityLevel[ObesityLevel]

def EncodeGenderValue(
        GenderValue: str,
    ) -> int:
    """
    Function for encoding 
    gender

    Parameter
    ---------
    GenderValue: str
        Gender value to encode

    Return
    ------
    Encoded_GenderValue: int
        Encoded gender value
    """

    return int(GenderValue[0] == 'M')

TransportationMethods = np.array(['Public_Transportation', 'Walking', 'Automobile', 'Motorbike','Bike'])
def EncodeMTransValue(
        MTransValue: str,
    ) -> np.ndarray:
    """
    Function for one hot 
    encoding of MTRANS values 

    Parameter
    ---------
    MTransValue: str
        MTRANS value to encode

    Return
    ------
    OneHotEncoded_MTransValue: np.ndarray
        OHE of MTRANS value
    """
    
    return (TransportationMethods == MTransValue).astype(int) 