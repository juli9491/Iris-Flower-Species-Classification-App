### Import required libraries ###
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle
import gzip


### Functions ###

### Model loading function ###
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



### Streamlit app design ###

### Set page configuration ###
st.set_page_config(page_title="Iris Flower Species Classification", page_icon="🌸", layout="centered")

### Set app title and description ###
st.title("🌸 Iris Flower Species Classification")

### App description ### 
st.info(
    """
    ℹ️ **This application uses AI 🤖 to predict the species of an Iris flower based on its characteristics. The application will predict the species of the Iris flower and display an image corresponding to the predicted species. Additionally, the application displays the prediction probabilities for each species to provide more insight into the prediction.**
    
    **📝 Instructions:**
    1. Adjust the sliders in the sidebar to input the characteristics of the Iris flower.
    2. The predicted species will be displayed.
    3. 👇Click on the expanders to see the predicted species image and prediction probabilities.
    """
)

### Load the trained model ###
model = load_model()

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
    prob_df = pd.DataFrame(probabilities, index=model.classes_, columns=["Probability"])
    st.bar_chart(prob_df)

### Additional information about the species in the sidebar ###
st.sidebar.subheader("Learn more about Iris species 💡")
st.sidebar.markdown("Hover over the species names to see more information.")
st.sidebar.markdown(
    """
    <div style="display: flex; align-items: center;">
        <span title="Iris Setosa: This species is known for its small size and bright blue or purple flowers.">🌸 Iris Setosa</span>
    </div>
    <div style="display: flex; align-items: center;">
        <span title="Iris Versicolor: Also known as the Blue Flag Iris, it has larger flowers that can be blue, purple, or yellow.">🌸 Iris Versicolor</span>
    </div>
    <div style="display: flex; align-items: center;">
        <span title="Iris Virginica: This species is similar to Iris Versicolor but typically has larger flowers and can be found in a variety of colors.">🌸 Iris Virginica</span>
    </div>
    """,
    unsafe_allow_html=True
)
