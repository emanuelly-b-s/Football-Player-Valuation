# Football-Player-Valuation

This project aimed to assist in machine learning education.

Problem Description:
A football team is restructuring their squad, and one of the planned actions is to purchase a goalkeeper. The team needs you to indicate the market value of this goalkeeper so 
that they can make a good deal. Using the provided dataset of football players, create at least three models capable of assisting the team in negotiating the player.

Dataset:
The project utilizes a dataset of football players, which includes various attributes such as age, performance statistics, market value, and other relevant information. 
This dataset will serve as the basis for training and evaluating the models.

Data Preprocessing:
The project involves several steps of data preprocessing:

Data cleaning: Removing any irrelevant columns or rows that are not related to goalkeepers.
Handling missing values: Removing or imputing missing values in the dataset.
Adjusting the market value format: Converting the market value column to a numeric format by removing currency symbols and converting the values to a standardized format (e.g., million dollars).

Model Development:
The project implements three different regression models to evaluate the goalkeeper:
Linear Regression: Using attributes such as "GKDiving," "GKHandling," "GKKicking," "GKReflexes," and "GKPositioning" to predict the market value.

Decision Tree Regressor: Building a decision tree-based model to estimate the goalkeeper's market value using the provided attributes.

Random Forest Regressor: Constructing an ensemble model using multiple decision trees to predict the market value accurately.
