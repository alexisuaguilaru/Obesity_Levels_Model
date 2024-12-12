# Model for estimating Obesity Levels
This project aims to develop a machine learning (ML) model to estimate the probability of suffering from a certain level of obesity based on dietary habits and physical condition.

The model uses clustering algorithms for estimating the probability of belonging to each type of obesity, reporting the most likely.

# Dataset Overview
[1] This dataset include data for the estimation of obesity levels in individuals from the countries of Mexico, Peru and Colombia, based on their eating habits and physical condition. The data contains 17 attributes and 2111 records, the records are labeled with the class variable NObesity (Obesity Level), that allows classification of the data using the values of Insufficient Weight, Normal Weight, Overweight Level I, Overweight Level II, Obesity Type I, Obesity Type II and Obesity Type III.

# Project Overview
## Exploratory Data Analysis (EDA)
An EDA was performed to determine and know some characteristics of the dataset to give a preliminary analysis of the behavior (accuracy, performance) of the model. 

## Model Overview
Three sub-models are used to estimate the probabilities of belonging to each type of obesity: the first sub-model estimates those of belonging to Insufficient Weight, Normal Weight, Overweight and Obesity; the second, those to type of Overweight; and the third, those to type of Obesity.

# References
- [1] Estimation of Obesity Levels Based On Eating Habits and Physical Condition, UCI Machine Learning Repository, DOI: https://doi.org/10.24432/C5H31Z, 2019.