# Model for estimating Obesity Levels
This project aims to develop a machine learning (ML) model to estimate the probability of suffering from a certain level of obesity based on dietary habits and physical condition.

The model uses clustering algorithms for estimating the probability of belonging to each type of obesity, reporting the most likely.

# Dataset Overview
[1] This dataset include data for the estimation of obesity levels in individuals from the countries of Mexico, Peru and Colombia, based on their eating habits and physical condition. The data contains 17 attributes and 2111 records, the records are labeled with the class variable NObesity (Obesity Level), that allows classification of the data using the values of Insufficient Weight, Normal Weight, Overweight Level I, Overweight Level II, Obesity Type I, Obesity Type II and Obesity Type III.

# Exploratory Data Analysis (EDA)
An EDA was performed to explore and discover some characteristics of the dataset's features and their correlation with the target feature (Obesity Level), as well as knowing the features' distributions and correlation with others features. This was performed in each sub dataset in which the original dataset was reorganized.

The justification of why using different sub-models could improve the accuracy of the final model was also shown, mainly based on how the correlation between the features and their relevance changed in each sub dataset. Therefore, this will allow to find representatives in each cluster with higher quality for containing the most relevant feature of a typical individual in some category (Obesity Level).

# Model Overview
Four sub-models are used to estimate the probabilities of belonging to each type of obesity: the first sub-model estimates those of belonging to Insufficient-Normal Weight, Overweight and Obesity; the second, those to Insufficient and Normal weight; the third, those to type of Overweight; and the fourth, those to type of Obesity.

# References
- [1] Estimation of Obesity Levels Based On Eating Habits and Physical Condition, UCI Machine Learning Repository, DOI: https://doi.org/10.24432/C5H31Z, 2019.