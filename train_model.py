#Car Price Prediction Using Random Forest

#Step 1 : Import Required Libraries

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#step 2 : Load dataset

data = pd.read_csv("car_data.csv")

#step 3 : Display first 5 rows

print(data.head())

#step 4 : Check missing values

print(data.isnull().sum())

#step 5 : Create New Feature - Car Age
#Assuming current year is 2026

data["Car_Age"] = 2026 - data["Year"]

#step 6 : Drop Unnecessary columns

data = data.drop(["Car_Name", "Year"], axis=1)

#step 7 : Convert categorical columns into numbers

data = pd.get_dummies(data, drop_first=True)

#step 8 : separate input and output

X = data.drop("Selling_Price",axis=1)

y = data["Selling_Price"]

#step 9 : Split dataset into training and testing data

X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.2, random_state=42
    )

#step 10 : create random forest model

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
    )

#step 11: Train the Model

model.fit(X_train,y_train)

#step 12 : predict on test data

y_pred = model.predict(X_test)

#step 13 : evaluate model

mae = mean_absolute_error(y_test,y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("MAE:",mae)
print("MSE:",mse)
print("RMSE:",rmse)
print("R2 Score:", r2)

#step 14 : Save trained model and feature names

joblib.dump(model, "car_price_model.pkl")
joblib.dump(X.columns, "model_features.pkl")

print("Model saved successfully!")

