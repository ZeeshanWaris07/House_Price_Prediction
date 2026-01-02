House Price Prediction
Project Overview

This project predicts house prices using property features such as area, number of bedrooms, bathrooms, stories, parking, and furnishing status. The model is trained on a dataset from Kaggle and uses linear regression with gradient descent to learn the relationship between features and house prices.

Features Used

Numeric Features: area, bedrooms, bathrooms, stories, parking

Categorical Features: furnishingstatus, hotwaterheating, airconditioning, prefarea, mainroad, basement, guestroom

Yes/No features are converted to 0/1.

furnishingstatus is one-hot encoded to allow the model to distinguish between categories.

Data Preprocessing

Outlier Handling: Prices are capped using the IQR method to reduce the effect of extremely high or low values.

Scaling: Numeric features and price are standardized using Z-score normalization.

Encoding: Categorical features are converted to numerical representations for model compatibility.

Model

Algorithm: Linear Regression

Optimization: Gradient Descent

Loss Metric: RMSE (Root Mean Squared Error)

Weights and bias are updated iteratively to minimize prediction error.

Evaluation

Metrics Used:

MAE (Mean Absolute Error)

RMSE (Root Mean Squared Error)

Visualizations are provided to compare predicted vs actual house prices.

Visualization

Scatter plots show how predicted prices match with actual prices, providing an intuitive understanding of model performance.

Usage

Clone the repository.

Place the Housing.csv dataset in the project folder.

Run house_price_prediction.py (or main script) to preprocess data, train the model, and visualize results.

Skills Demonstrated

Regression modeling and gradient descent

Feature preprocessing: scaling, one-hot encoding, handling categorical data

Outlier detection and handling (IQR method)

Model evaluation using MAE and RMSE

Data visualization with scatter plots
