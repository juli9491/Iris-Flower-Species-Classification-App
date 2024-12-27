### Import required libraries ###
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle
import gzip
#import os

### Cache the model loading function ###
@st.cache_resource
def load_model():
    with gzip.open("Model.pkl.gz", 'rb') as file:
        model = pickle.load(file)
    return model

### Cache the prediction function ###
def predict_species(model, input_df):
    return model.predict(input_df)[0]

### Set the working directory to the specified path ###
#os.chdir("C:\\Users\\Fateh-Nassim MELZI\\Documents\\AI_Projects\\Iris_Flower_Spicies_Classification_Project\\App_Construction")

### Load the trained model ###
model = load_model()

### Streamlit app design ###
st.sidebar.header('What is the species of my Iris flower?')
st.sidebar.subheader("Iris flower characteristics:")

### Set up sliders for each characteristic of the iris flower ###
sepal_length = st.sidebar.slider("Sepal length", 4.3, 7.9, 5.3)
sepal_width = st.sidebar.slider("Sepal width", 2.0, 4.4, 3.3)
petal_length = st.sidebar.slider("Petal length", 1.0, 6.9, 2.3)
petal_width = st.sidebar.slider("Petal width", 0.1, 2.5, 1.3)

### Retrieve the characteristics of the iris flower as a dictionary ###
input_dict = {'sepal_length': sepal_length, 'sepal_width': sepal_width, 'petal_length': petal_length, 'petal_width': petal_width}

### Convert the dictionary into a DataFrame ###
input_df = pd.DataFrame(input_dict, index=[0])

### Predict the species of the iris flower based on the entered characteristics ###
iris_species = predict_species(model, input_df)

### Display the predicted species of the iris flower ###
st.header(f"The iris flower species is: {iris_species}")

if iris_species == 'versicolor':
    st.image("Iris_Versicolor_Image.png", caption="", use_container_width=True)
elif iris_species == 'virginica':
    st.image("Iris_Virginica_Image.png", caption="", use_container_width=True)
else:
    st.image("Iris_Setosa_Image.png", caption="", use_container_width=True)
