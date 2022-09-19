import streamlit as st
from streamlit_lottie import st_lottie
import pandas as pd
import numpy as np
import joblib
from preprocessing import preprocess
import requests

st.set_page_config(page_title="Churn Predictor", page_icon="💻", layout="wide")

# Define a function that we can use to load lottie files from a link.
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie = load_lottieurl("https://assets8.lottiefiles.com/packages/lf20_3rqwsqnj.json")
model = joblib.load("model.sav")



def main():
    col1, col2 = st.columns([1, 3])
    with col1:
        st_lottie(lottie, width=300, height=300)
    
    with col2:
        st.title("Telco Customer Churn Prediction App")
        st.write("This Streamlit app is made to predict customer churn in a fictional telecommunication use case.")
        st.write("The application is functional for both online prediction and batch data prediction.")

    # Set up the sidebar
    add_selectbox = st.selectbox(
        "Select the type of prediction", ("Online", "Batch"))
    
    if add_selectbox == "Online":
        st.write("#### Online Prediction")
        st.info("Please fill in the following details to get a prediction.")
        # Set up the form
        st.subheader("Demographic Information")
        seniorCitizen = st.selectbox("Senior Citizen", ("Yes", "No"))
        dependents = st.selectbox("Dependent", ("Yes", "No"))

        st.subheader("Payment Information")
        tenure = st.slider("Number of months the customer has stayed with the company", min_value=0, max_value=72)
        contract = st.selectbox("Contract type", ("Month-to-month", "One year", "Two year"))
        paperlessBilling = st.selectbox("Paperless billing", ("Yes", "No"))
        paymentMethod = st.selectbox("Payment method", ("Electronic check", "Mailed check", "Bank transfer (automatic)",  "Credit card (automatic)"))
        monthlyCharges = st.number_input("Monthly charges", min_value=0, max_value=150)
        totalCharges = st.number_input("Total charges", min_value=0, max_value=10000)

        st.subheader("Services Singed Up For")
        multipleLines = st.selectbox("Does the customer have multiple lines", ("Yes", "No", "No phone service"))
        phoneService = st.selectbox("Does the customer have phone service", ("Yes", "No"))
        internetService = st.selectbox("Does the customer have internet service", ("DSL", "Fiber optic", "No"))
        onlineSecurity = st.selectbox("Does the customer have online security service", ("Yes", "No", "No internet service"))
        onlineBackup = st.selectbox("Does the customer have online backup service", ("Yes", "No", "No internet service"))
        techSupport = st.selectbox("Does the customer have tech support service", ("Yes", "No", "No internet service"))
        streamingTv = st.selectbox("Does the customer have streaming TV service", ("Yes", "No", "No internet service"))
        streamingMovies = st.selectbox("Does the customer have streaming movies service", ("Yes", "No", "No internet service"))

        # Assign the values to a dictionary
        data = {"SeniorCitizen": seniorCitizen,
                "Dependents": dependents,
                "tenure": tenure,
                "Contract": contract,
                "PaperlessBilling": paperlessBilling,
                "PaymentMethod": paymentMethod,
                "MonthlyCharges": monthlyCharges,
                "TotalCharges": totalCharges,
                "MultipleLines": multipleLines,
                "PhoneService": phoneService,
                "InternetService": internetService,
                "OnlineSecurity": onlineSecurity,
                "OnlineBackup": onlineBackup,
                "TechSupport": techSupport,
                "StreamingTV": streamingTv,
                "StreamingMovies": streamingMovies}
        
        features_df = pd.DataFrame.from_dict([data])
        st.write("Overview of input is shown below")
        st.dataframe(features_df)

        # Preprocess the input data
        preprocess_df = preprocess(features_df, "Online")

        # Predict the churn
        prediction = model.predict(preprocess_df)

        if st.button("Predict"):
            if prediction == 1:
                st.warning("The customer is likely to churn.")
            else:
                st.success("The customer seems satisfied with the services.")

    elif add_selectbox == "Batch":
        st.write("#### Batch Prediction")
        st.write("You can use this [sample dataset](https://github.com/BatuhanYilmaz26/Churn-Predictor/blob/master/data/batch_churn.csv) to experiment with the batch prediction.")
        file_upload = st.file_uploader("Please upload the CSV file to get predictions.", type=["csv"])
        if file_upload is not None:
            df = pd.read_csv(file_upload)
            st.dataframe(df)

            # Preprocess the input data
            preprocess_df = preprocess(df, "Batch")

            if st.button("Predict"):
                # Get batch predictions
                predictions = model.predict(preprocess_df)
                prediction_df = pd.DataFrame(predictions, columns=["Predictions"])
                prediction_df = prediction_df.replace({1: "The customer is likely to churn.",
                                                       0: "The customer seems satisfied with the services."})
                st.dataframe(prediction_df)


if __name__ == "__main__":
        main()