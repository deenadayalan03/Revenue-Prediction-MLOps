import os 
import numpy as np
import pandas as pd
from mlproject.pipeline.prediction import PredictionPipeline
import streamlit as st

model = PredictionPipeline()

def revenue_prediction(input_data):
    
    # Ensure input data is numeric
    input_data_as_numpy_array = np.asarray(input_data, dtype=float)
    
    # Reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    print(input_data_reshaped)
    
    # Make prediction
    prediction = model.predict(input_data_reshaped)
    
    return prediction

def main():
    st.title('Revenue Prediction App')
    
    Product_Category = st.selectbox('Select the ProductCategory', ['Electronics', 'Home Appliances','Clothing','Books','Beauty Products','Sports'])
    Region = st.selectbox('Slect your Region', ['North America', 'Europe','Asia'])
    Payment_Method = st.selectbox('Select the Payment Method', ['Credit Card', 'PayPal','Debit Card'])
    Unit_Sold = st.number_input('Enter no of Unit_Sold', min_value=0)
    Unit_Price = st.number_input('Enther the Unit_Price', min_value=0)
    Month = st.selectbox('Enter the Month', ['January', 'March', 'May', 'July', 'April', 'June','February','August'])

    Product_Category_mapping = {'Electronics':3,'Home Appliances':4,'Clothing':2,'Books':1,'Beauty Products':0,'Sports':5}
    Region_mapping = {'North America':2, 'Europe':1,'Asia':0}
    Payment_Method_mapping = {'Credit Card':0, 'PayPal':2,'Debit Card':1}
    Month_mapping = {'January':1, 'March':3, 'May':5, 'July':7, 'April':4, 'June':5,'February':2,'August':8}
    
    Product_Category = Product_Category_mapping[Product_Category]
    Region = Region_mapping[Region]
    Payment_Method = Payment_Method_mapping[Payment_Method]
    Month = Month_mapping[Month]
    
    
    # Code for prediction
    Prediction = ''

    # Creating a button for prediction
    if st.button('Predict Revenue'):
        Prediction = revenue_prediction([Product_Category, Region, Payment_Method,  Unit_Sold , Unit_Price, Month])
    
    st.success(Prediction)

if __name__ == '__main__':
    main()



































