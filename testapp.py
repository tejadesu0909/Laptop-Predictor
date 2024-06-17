import streamlit as st
import pickle
import numpy as np
from scipy.special import inv_boxcox


pipe = pickle.load(open('pipe.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))
st.title("Laptop Predictor")

# brand
company = st.selectbox('Brand', df['Company'].unique())

type = st.selectbox('Type', df['TypeName'].unique())

ram = st.selectbox('Ram', df['Ram'].unique())

opSys = st.selectbox('OpSys', df['OpSys'].unique())

# weight
weight = st.number_input('Weight of the Laptop')

Cpu_Clock_Speed = st.number_input('Cpu Clock Speed of the Laptop')

Cpu_brand = st.selectbox('Type', df['Cpu_brand'].unique())


Cpu_processor = st.selectbox('Type', df['Cpu_processor'].unique())


HDD = st.number_input('HDD Storage of the Laptop')

SSD = st.number_input('SSD Storage of the Laptop')

FlashStorage = st.number_input('FlashStorage of the Laptop')

Hybrid = st.number_input('Hybrid Storage of the Laptop')

PPI = st.number_input('PPI Pixel Per Inch of the Laptop')

				
Gpu_brand = st.selectbox('Type', df['Gpu_brand'].unique())

if st.button('Predict Price'):
    query = np.array([company, type, ram, opSys, weight, Cpu_Clock_Speed, Cpu_brand,	Cpu_processor,	HDD,	SSD,	FlashStorage,	Hybrid,	PPI,	Gpu_brand])
    query = query.reshape(1, 14)
    # predicted_price = np.exp(pipe.predict(query)) 
    predicted_price_bc = pipe.predict(query)
    # using this 0.11085425321845482 as the lambda value that i had used to transform the target value using boxcox transformation. 
    # predicted_price = inv_boxcox(predicted_price_bc, 0.lambada_bc)

    predicted_price = inv_boxcox(predicted_price_bc, 0.11085425321845482)

    st.title(f"Predicted Laptop Price: ${predicted_price[0]:.2f}")




