import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from gradcam import GradCAM
import cv2

st.set_option('deprecation.showfileUploaderEncoding', False)

st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Select a page:", ["Dataset", "Prediction"], key="navigation_selectbox")

model_name = st.sidebar.selectbox("Select a Model:", ["Credit Scoring Model 1", "Credit Scoring Model 2", "Credit Scoring Model 3"], key="model_selectbox")


class CreditScoringModel(nn.Module):
    def __init__(self, input_dim=7):
        super(CreditScoringModel, self).__init__()
        self.layer1 = nn.Linear(input_dim, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.layer2 = nn.Linear(64, 32)
        self.layer3 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        print("Input shape:", x.shape)
        x = self.layer1(x)
        print("Weight shape:", self.layer1.weight.shape)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer3(x)
        x = self.sigmoid(x)
        return x

class CreditScoringModel2(nn.Module):
    def __init__(self, input_dim=7):
        super(CreditScoringModel2, self).__init__()
        self.layer1 = nn.Linear(input_dim, 64)
        self.relu = nn.LeakyReLU()  # Change ReLU to LeakyReLU
        self.dropout = nn.Dropout(0.2)
        self.layer2 = nn.Linear(64, 32)
        self.layer3 = nn.Linear(32, 1)
        self.sigmoid = nn.Tanh()  # Change sigmoid to tanh

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer3(x)
        x = self.sigmoid(x)  # Use tanh activation function instead of sigmoid
        return x

class CreditScoringModel3(nn.Module):
    def __init__(self, input_dim):
        super(CreditScoringModel3, self).__init__()
        self.layer1 = nn.Linear(input_dim, 64)  # Change 128 to 64
        self.relu = nn.LeakyReLU()  # Change ReLU to LeakyReLU
        self.dropout = nn.Dropout(0.2)
        self.layer2 = nn.Linear(64, 32)  # Change 128 to 64
        self.layer3 = nn.Linear(32, 1)  # Change 64 to 32
        # Remove layer4
        self.tanh = nn.Tanh()  # Change sigmoid to tanh

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer3(x)
        # Remove relu and dropout layers
        x = self.tanh(x)  # Use tanh activation function instead of sigmoid
        return x




def load_model(model_name):
    if model_name == "Credit Scoring Model 1":
        model = CreditScoringModel(X_train.shape[1])
        model.load_state_dict(torch.load("credit_risk_model_1.pth"))
    elif model_name == "Credit Scoring Model 2":
        model = CreditScoringModel2(X_train.shape[1])
        model.load_state_dict(torch.load("credit_risk_model_2.pth"))
    elif model_name == "Credit Scoring Model 3":
        model = CreditScoringModel3(X_train.shape[1])
        model.load_state_dict(torch.load("credit_risk_model_2.pth"))    

    model.eval()
    return model


def create_one_hot_encoded_user_input(user_input, X):
    dummy_columns = [col for col in X.columns if col not in user_input.columns]
    for col in dummy_columns:
        user_input[col] = 0

    user_input = user_input[X.columns]
    return user_input


def load_data(url):
    column_names = ['checking_account', 'duration', 'credit_history', 'purpose', 'credit_amount',
                    'savings_account', 'employment', 'installment_rate', 'personal_status_sex',
                    'debtors_guarantors', 'residence', 'property', 'age', 'other_installment_plans',
                    'housing', 'credits', 'job', 'liable_people', 'telephone', 'foreign_worker', 'credit_risk']
    data = pd.read_csv(url, names=column_names, delimiter=' ')

    # Preprocess the categorical variables using one-hot encoding
    data = pd.get_dummies(data, columns=['checking_account', 'credit_history', 'purpose', 'savings_account', 'employment',
                                         'personal_status_sex', 'debtors_guarantors', 'property', 'other_installment_plans',
                                         'housing', 'job', 'telephone', 'foreign_worker'])

    # Split the data into features (X) and target (y)
    X = data.drop(columns=['credit_risk'])
    y = data['credit_risk'].map({1: 0, 2: 1})  # Map the target variable to 0 (good risk) and 1 (bad risk)

    return X, y

def preprocess_data(X, y):
    # Standardize the continuous features
    continuous_features = ['duration', 'credit_amount', 'installment_rate', 'residence', 'age', 'credits', 'liable_people']

    scaler = StandardScaler()
    X[continuous_features] = scaler.fit_transform(X[continuous_features])
    joblib.dump(scaler, 'scaler.pkl')  # Save the scaler object to a file

    # Check for missing values in the dataset
    missing_values = X.isnull().sum()
    print("Missing values in the dataset:\n", missing_values)

    # Fill missing values with the mean, if any
    X.fillna(X.mean(), inplace=True)
    print("Data types in the dataset:\n", X.dtypes)

    # Convert boolean columns to integers
    bool_columns = X.select_dtypes(include=[bool]).columns
    X[bool_columns] = X[bool_columns].astype(int)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, scaler




if page == "Dataset":
    st.title("German Credit Dataset")
    st.write("""
    This app uses a neural network to predict credit risk based on German credit data.
    """)

    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
    X, y = load_data(url)
    X_train, X_test, y_train, y_test, scaler = preprocess_data(X, y)
    st.write("Data sample:")
    st.write(pd.concat([X, y], axis=1).head())

elif page == "Prediction":
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
    X, y = load_data(url)
    X_train, X_test, y_train, y_test, scaler = preprocess_data(X, y)
    continuous_features = ['duration', 'credit_amount', 'installment_rate', 'residence', 'age', 'credits', 'liable_people']
    scaler = joblib.load('scaler.pkl')  # Load the scaler object from the file

    st.title("Credit Risk Prediction")
    st.write("""
    Input some values and the model will predict if the credit risk is Good or Bad.
    """)

    # Create input fields for user
    duration = st.number_input("Duration (in months):", min_value=1, max_value=120, value=12, step=1)
    credit_amount = st.number_input("Credit Amount:", min_value=100, max_value=20000, value=1000, step=100)
    age = st.number_input("Age (in years):", min_value=18, max_value=120, value=30, step=1)
    installment_rate = st.number_input("Installment Rate (in percentage of income):", min_value=1, max_value=4, value=2, step=1)
    residence = st.number_input("Years at current residence:", min_value=1, max_value=4, value=2, step=1)
    credits = st.number_input("Number of existing credits:", min_value=1, max_value=5, value=1, step=1)
    liable_people = st.number_input("Number of people being liable to provide maintenance:", min_value=1, max_value=2, value=1, step=1)
    
    # Get the user input as a dataframe
    user_input = pd.DataFrame({
        'duration': [duration],
        'credit_amount': [credit_amount],
        'installment_rate': [installment_rate],
        'residence': [residence],
        'age': [age],
        'credits': [credits],
        'liable_people': [liable_people]
    })

        # Instantiate the model
    #model = CreditScoringModel(X_train.shape[1])

    model = load_model(model_name)
    # Load the model's state_dict
    #model.load_state_dict(torch.load("credit_risk_model.pth"))

    # Set the model to evaluation mode
    model.eval()


    # Get the user input as a dataframe
    user_input = pd.DataFrame({
        'duration': [duration],
        'credit_amount': [credit_amount],
        'installment_rate': [installment_rate],
        'residence': [residence],
        'age': [age],
        'credits': [credits],
        'liable_people': [liable_people]
    })

    # One-hot encode the categorical variables in the user input
    user_input = create_one_hot_encoded_user_input(user_input, X_train)

    if st.button("Predict"):
        # Preprocess and standardize the user input
        user_input[continuous_features] = scaler.transform(user_input[continuous_features])

        # Convert the input into a PyTorch tensor
        user_input_tensor = torch.tensor(user_input.values, dtype=torch.float32)

        # Make a prediction using the trained model
        prediction = model(user_input_tensor)
        prediction = (prediction.detach().numpy() > 0.5).astype(int)

        # Display the prediction result
        if int(prediction) == 0:
            st.write("Predicted Credit Risk: Good")
        else:
            st.write("Predicted Credit Risk: Bad")









    
                





