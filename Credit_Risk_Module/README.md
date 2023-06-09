# Credit Risk Prediction App:

This repository contains a Streamlit web application that uses a neural network to predict credit risk based on German credit data. The app provides an interactive interface for users to input their data and receive a prediction on whether their credit risk is "Good" or "Bad".

## Dataset
The dataset used for this project is the German Credit dataset, which contains 1000 instances with 20 attributes and a binary target variable indicating credit risk. The attributes include both continuous and categorical features, such as duration, credit amount, age, and checking account status.

## Models 
Three different neural network models are available for selection in the app:

Credit Scoring Model 1
Credit Scoring Model 2
Credit Scoring Model 3
These models have different architectures and activation functions, allowing users to explore the impact of these choices on the prediction results.

## Installation
1. Clone the repository

2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # For Windows, use "venv\Scripts\activate"

3. Install the required packages:
pip install -r requirements.txt

4. Run the Streamlit app
streamlit run app.py

5. Open the app in your web browser by navigating to the URL displayed in the terminal (usually http://localhost:8501).

## Usage 

1. Select a model from the sidebar.
2. Navigate to the "Prediction" page.
3. Input your data using the provided input fields.
4. Click the "Predict" button to generate a prediction.
5. The app will display the predicted credit risk as either "Good" or "Bad" based on the selected model and input data.

## Diagram

<img width="1792" alt="image" src="https://user-images.githubusercontent.com/111610085/236722432-61caf332-93a5-4911-aa69-d14730138e5c.png">

