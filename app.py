import numpy as np
import streamlit as st
import pickle

# Importing the model
scaled = pickle.load((open("scalled_data1.pkl",'rb')))
main_df = pickle.load(open("diamond_df.pkl",'rb'))

st.title("Diamond Price Predictor")

# Imformation
# carat weight of the diamond (0.2--5.01)
caret = st.selectbox("CARET (carat weight of the diamond)",main_df['carat'].unique())

#cut
# quality of the cut (Fair, Good, Very Good, Premium, Ideal)
cut = st.selectbox('CUT (quality of the cut)',['Fair', 'Good', 'Very Good','Premium', 'Ideal'])

#color
#diamond colour, from J (worst) to D (best)
color = st.selectbox("COLOR (diamond colour, from J (worst) to D (best))",['D','E','F','G','H','I','J'])

#clarity
#a measurement of how clear the diamond is (I1 (worst), SI2, SI1, VS2, VS1, VVS2, VVS1, IF (best))
clarity = st.selectbox("CLARITY (a measurement of how clear the diamond is (I1 (worst), SI2, SI1, VS2, VS1, VVS2, VVS1, IF (best))",['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF'])

#X,Y,Z

x = st.selectbox("X",main_df['x'].unique())
y = st.selectbox("Y",main_df['y'].unique())
z = st.selectbox("Z",main_df['z'].unique())

if st.button("Predict Price"):
    quary = np.array([caret,cut,color,clarity,x,y,z])
    quary = quary.reshape((1,7))
    st.title(scaled.predict(quary))