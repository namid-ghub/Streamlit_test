
"""'''
import numpy as np
import streamlit as st
import pandas as pd

import pickle  # to load a saved model
import base64  # to handle gif encoding

@st.cache_data
def get_fvalue(val):
    feature_dict = {"No": 1, "Yes": 2}
    return feature_dict[val]

def get_value(val, my_dict):    
    return my_dict[val]

# Define app_mode using a Streamlit selectbox
app_mode = st.sidebar.selectbox('Choose the app mode', ['Home', 'Predict', 'About'])

if app_mode == 'Home':    
    st.title('Loan Prediction')    
    st.image('chart_t1.png')    
    st.markdown('Dataset:')    
    data = pd.read_csv('loan_data.csv')    
    st.write(data.head())    
    #st.bar_chart(data[['income_annum', 'loan_amount']].head(20))
    st.bar_chart(data[['income_annum', 'loan_amount']].head(20))  
    print(data.columns) # to see the columns in the terminal
    st.write(data.columns) #to see the columns in the browser


if app_mode == 'Prediction':    
    ApplicantIncome = st.sidebar.slider('ApplicantIncome', 0, 10000, 0)    
    LoanAmount = st.sidebar.slider('LoanAmount in K$', 9.0, 700.0, 200.0)    # Assuming additional input features here...    
    
    # Prediction Logic   
if st.button("Predict"):
    loaded_model = pickle.load(open('Random_Forest.sav', 'rb'))        
    prediction = loaded_model.predict(np.array([ApplicantIncome, LoanAmount]).reshape(1, -1))        

if prediction[0] == 0:  
    st.error('According to our calculations, you will not get the loan.') 
else:            
    st.success('Congratulations! You will get the loan.')

    """

import numpy as np
import streamlit as st
import pandas as pd

import pickle  # to load a saved model
import base64  # to handle gif encoding

@st.cache_data
def get_fvalue(val):
    feature_dict = {"No": 1, "Yes": 2}
    return feature_dict[val]

def get_value(val, my_dict):    
    return my_dict[val]

# Define app_mode using a Streamlit selectbox
app_mode = st.sidebar.selectbox('Choose the app mode', ['Home', 'Predict', 'About'])

if app_mode == 'Home':    
    st.title('Loan Prediction')    
    st.image('chart_t1.png')    
    st.markdown('Dataset:')    
    data = pd.read_csv('loan_data.csv')    
    st.write(data.head())    
    st.bar_chart(data[['income_annum', 'loan_amount']].head(20))  
    print(data.columns)  # Terminal output
    st.write(data.columns)  # Browser output

elif app_mode == 'Predict':    
    st.title("Loan Eligibility Predictor")
    
    # Sidebar inputs
    ApplicantIncome = st.sidebar.slider('Applicant Income ($)', 0, 10000, 3000)
    LoanAmount = st.sidebar.slider('Loan Amount ($K)', 9.0, 700.0, 200.0)
    
    # Add more inputs if needed, e.g., credit history, gender, etc.

    if st.button("Predict"):
        try:
            # Load the trained model
            loaded_model = pickle.load(open('Random_Forest.sav', 'rb'))

            # Make prediction
            input_data = np.array([ApplicantIncome, LoanAmount]).reshape(1, -1)
            prediction = loaded_model.predict(input_data)

            # Show result
            if prediction[0] == 0:
                st.error('According to our calculations, you will NOT get the loan.')
            else:
                st.success('Congratulations! You are likely to get the loan.')
        
        except FileNotFoundError:
            st.error("Model file not found. Make sure 'Random_Forest.sav' exists in the app directory.")

elif app_mode == 'About':
    st.title("About")
    st.markdown("""
    This Streamlit app predicts loan eligibility based on applicant income and loan amount.
    
    - Built with Python and Streamlit  
    - Machine Learning model: Random Forest  
    - Dataset: `loan_data.csv`
    """)


