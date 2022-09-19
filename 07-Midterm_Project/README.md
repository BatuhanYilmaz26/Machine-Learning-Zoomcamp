## Churn-Predictor
![Python](https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=blue)
![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Heroku](https://img.shields.io/badge/heroku-%23430098.svg?style=for-the-badge&logo=heroku&logoColor=white)

#### About this project
- This project is a customer churn prediction system built using the [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).
- The dataset contains information about customer demographics, payment details and services that the customer signed up for.
- Trained and evaluated various machine learning models such as Logistic Regression, SVC, Random Forest Classifier, Decision Tree Classifier, XGBoost Classifier, LightGBM Classifier. 
- Performed feature selection using Recursive Feature Elimination technique and tuned the hyperparameters for the best performing model which was Logistic Regression model.
- Saved the best model using [joblib library](https://joblib.readthedocs.io/en/latest/) and used it to build an interactive web application.
- Built the web app using [Streamlit](https://streamlit.io) and deployed it on [Heroku](https://www.heroku.com).
- You can take a look at the [interactive demo](https://churn-predictorx.herokuapp.com) to see how the customer churn prediction system works.
  - You can get online predictions by filling out the form in the web application.
  - You can use this [sample dataset](https://github.com/BatuhanYilmaz26/Churn-Predictor/blob/master/data/batch_churn.csv) to experiment with the batch prediction.
