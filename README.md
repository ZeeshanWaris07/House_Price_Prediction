House Price Prediction
Project Overview

This project predicts house prices using property features such as area, number of bedrooms, bathrooms, stories, parking, and furnishing status. The model is trained on a dataset from Kaggle and uses linear regression with gradient descent to learn the relationship between features and house prices.

Features Used

Numeric Features: area, bedrooms, bathrooms, stories, parking

Categorical Features: furnishingstatus, hotwaterheating, airconditioning, prefarea, mainroad, basement, guestroom

Yes/No features are converted to 0/1.


Data Preprocessing

Outlier Handling: Prices are capped using the IQR method to reduce the effect of extremely high or low values.

Scaling: Numeric features and price are standardized using Z-score normalization.

Encoding: Categorical features are converted to numerical representations for model compatibility.

Model

Algorithm: Linear Regression

Optimization: Gradient Descent

Loss Metric: RMSE (Root Mean Squared Error)

Weights and bias are updated iteratively to minimize prediction error.


Weights and Bias: 
Weights after training: [0.28930721122195047 0.04543014313426616 0.24937899867566074
 0.2231930511450057 0.22965690664074803 0.2053013168023436
 0.1886305737420166 0.43400146164114506 0.46967857738322155
 0.11455351038086833 0.3336192591635505 -0.019105874773175855
 -0.24122963707682285]
Bias after training: -0.4576251721156534


Evaluation

Metrics Used:

RMSE (Root Mean Squared Error)

Visualizations are provided to compare predicted vs actual house prices.

Visualization

Scatter plots show how predicted prices match with actual prices, providing an intuitive understanding of model performance.

Usage

Clone the repository.

Place the Housing.csv dataset in the project folder.

Run house_price_prediction.py (or main script) to preprocess data, train the model, and visualize results.

