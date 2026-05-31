# 🚗 Car Price Prediction using Random Forest

A beginner-friendly machine learning project to predict the selling price of used cars using the ""Cardekho dataset"" and the **Random Forest Regressor** algorithm.  

This project demonstrates the ""end-to-end workflow"" of a data science project:
- Data preprocessing (cleaning, feature engineering, encoding categorical variables)
- Model training and evaluation
- Saving and reusing trained models
- Deployment with ""Streamlit"" for real-time predictions

---

## 📂 Folder Structure

car_price_project/
│
├── car_data.csv              # Dataset (CarDekho)
├── train_model.py            # Model training script
├── app.py                    # Streamlit app for prediction
├── requirements.txt          # Dependencies
├── car_price_model.pkl       # Saved trained model
├── model_features.pkl        # Saved feature names
├── README.md                 # Project documentation
└── .gitignore                # Ignore unnecessary files



---

## 📊 Dataset

- Source: [CarDekho Vehicle Dataset](https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho)
- Important columns:
  - `Year`
  - `Present_Price`
  - `Kms_Driven`
  - `Fuel_Type`
  - `Seller_Type`
  - `Transmission`
  - `Owner`
- Target variable: **Selling_Price**

---

## ⚙️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/tigayalakshit7-stack/car-price-prediction.git
   cd car-price-prediction

2. Install Dependencies:

   pip install -r requirements.txt


## ▶️ Usage

1. Train the Model
    
   python train_model.py

2. Run the Streamlit App

   streamlit run app.py


## 📊 Example Input & Output


Input:

Year: 2018

Present Price: 8.5 lakh

Kms Driven: 35,000

Fuel Type: Petrol

Transmission: Manual

Output:

Predicted Car Price: ₹5.80 lakh

## 📈 Evaluation Metrics

:- MAE (Mean Absolute Error) -> Average prediction error
:- MSE (Mean Squared Error) -> Penalizes larger mistakes
:- RMSE (Root Mean Squared Error) -> Error in same unit as price
:- R²Score -> How well the model explains variance in data

## 📦 Dependencies 

* pandas
* numPy
* scikit-learn
* streamlit
* joblib

## 🎯 Project Outcome 

A machine learning model that predicts the estimated selling price of a used car with good accuracy, deployed as an interactive web app.



