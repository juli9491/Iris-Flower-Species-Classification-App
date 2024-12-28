### Import required libraries ###
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle
import gzip

### Cache the model loading function ###
@st.cache_resource
def load_model():
    with gzip.open("Model.pkl.gz", 'rb') as file:
        model = pickle.load(file)
    return model

### Prediction function ###
def predict_species(model, input_df):
    probabilities = model.predict_proba(input_df)[0]
    predicted_class = model.predict(input_df)[0]
    return predicted_class, probabilities

### Load the trained model ###
model = load_model()

### Streamlit app design ###
st.title("🌸Iris Flower Species Classification")

# Description with info box and emoji
st.info(
    """
    ℹ️ **This application allows users to predict the species of an Iris flower based on its characteristics. The application will predict the species of the Iris flower and display an image corresponding to the predicted species. Additionally, the application displays the prediction probabilities for each species to provide more insight into the prediction.**
    """
)

### Sidebar for user input ###
st.sidebar.header('Input Iris Flower Characteristics')
st.sidebar.subheader("Adjust the sliders to input the characteristics:")

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
iris_species, probabilities = predict_species(model, input_df)

### Display the predicted species of the iris flower ###
st.subheader(f"The predicted iris flower species is: 🌸 {iris_species}")

### Display corresponding image in an expander ###
with st.expander("See predicted species image"):
    if iris_species == 'versicolor':
        st.image("Iris_Versicolor_Image.png", caption="", use_container_width=True)
    elif iris_species == 'virginica':
        st.image("Iris_Virginica_Image.png", caption="", use_container_width=True)
    else:
        st.image("Iris_Setosa_Image.png", caption="", use_container_width=True)

### Display the prediction probabilities in an expander ###
with st.expander("See prediction probabilities"):
    #st.subheader("Prediction Probabilities")
    prob_df = pd.DataFrame(probabilities, index=model.classes_, columns=["Probability"])
    st.bar_chart(prob_df)
