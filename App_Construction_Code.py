### Import required libraries ###
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
from io import BytesIO

# Constants
TEMPLATE_PATH = "Template.xlsx"

### Functions ###
### Model loading function ###
@st.cache_resource
def load_model():
    try:
        return joblib.load("Model.pkl.xz")
    except FileNotFoundError:
        st.error("Model file not found. Please ensure 'Model.pkl.xz' is in the correct directory.")
        return None

### Prediction function ###
def predict_species(model, input_df):
    # Obtenir les probabilités pour toutes les lignes
    probabilities = model.predict_proba(input_df)
    
    # Obtenir les classes prédites pour toutes les lignes
    predicted_class = model.predict(input_df)
    
    return predicted_class, probabilities

# Load the pre-prepared Excel template
def load_template(file_path: str) -> bytes:
    with open(file_path, "rb") as file:
        return file.read()

def to_excel_data(data: pd.DataFrame, index: bool = False) -> bytes:
    template_io = BytesIO()
    data.to_excel(template_io, index=index)
    template_io.seek(0, 0)
    return template_io.read()


### Streamlit app design ###

### Set page configuration ###
st.set_page_config(page_title="Iris Flower Species Classification", page_icon="🌸", layout="centered")


### Set app title and description ###
st.header("🌸 Iris Flower Species Classification")


# Add explanation and image to the sidebar
st.sidebar.title("ℹ️ About:")
st.sidebar.write("""
This application uses AI 🤖 to predict the species of an Iris flower 🌸  based on its characteristics. The application will predict the species of the Iris flower and display an image corresponding to the predicted species. Additionally, the application displays the prediction probabilities 📊 for each species to provide more insight into the prediction.
""")

st.sidebar.image("Iris_Flower_Image.jpg", caption="", use_container_width=True)

### Load the trained model ###
model = load_model()

# Initialize session state for user input
if 'user_input' not in st.session_state:
    st.session_state.user_input = {}

# Create a single radio button group for prediction mode
prediction_mode = st.radio("Choose the classification mode:", ("One flower", "Several flowers"), index=0, horizontal=True)

# If the prediction mode is "One part"
if prediction_mode == "One flower":
    st.write("Adjust the sliders 👇 to input the characteristics:") 

    # Set up sliders for each characteristic of the iris flower
    sepal_length = st.slider("Sepal length", 4.3, 7.9, 5.3)
    sepal_width = st.slider("Sepal width", 2.0, 4.4, 3.3)
    petal_length = st.slider("Petal length", 1.0, 6.9, 2.3)
    petal_width = st.slider("Petal width", 0.1, 2.5, 1.3)

    # Create a DataFrame for the input
    st.session_state.user_input = {'sepal_length': sepal_length, 'sepal_width': sepal_width, 'petal_length': petal_length, 'petal_width': petal_width}
    input_df = pd.DataFrame(st.session_state.user_input, index=[0])

    # Predict the species and probabilities
    iris_species, probabilities = predict_species(model, input_df)

    # Display the predicted species
    st.success(f"The predicted iris flower 🌸 species is: {iris_species[0]}")  # Note: Access the first item for single prediction

    # Display corresponding image in an expander
    with st.expander("See predicted species image"):
        if iris_species[0] == 'versicolor':
            st.image("Iris_Versicolor_Image.png", caption="", use_container_width=True)
        elif iris_species[0] == 'virginica':
            st.image("Iris_Virginica_Image.png", caption="", use_container_width=True)
        else:
            st.image("Iris_Setosa_Image.png", caption="", use_container_width=True)

    # Display the prediction probabilities in an expander
    with st.expander("See prediction probabilities"):
        prob_df = pd.DataFrame(probabilities, columns=model.classes_)  # Create DataFrame with class names as columns
        st.bar_chart(prob_df.T)  # Transpose to align with expected format for a bar chart

    # If the classification mode is "Several flowers"
else:
    cols = st.columns(2)
    template = load_template(TEMPLATE_PATH)
    cols[0].download_button(
        'Download a template',
        template,
        file_name='template.xlsx',
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        help='Download an Excel file template'
    )
    
    st.info("""
    ⚠️ **Note:** please ensure the uploaded file follows the template structure with correct headers. This will avoid prediction errors.
    """)
    
    uploaded_file = st.file_uploader(
        label="Upload a file with flower characteristics",
        type=["csv", "xls", "xlsx"],
        accept_multiple_files=False,
        key="pred_flowers",
        help="Click on Browse to upload your file"
    )
    
    if uploaded_file is not None:
        try:
            # Read the uploaded file
            if uploaded_file.name.endswith('.csv'):
                flowers_df = pd.read_csv(uploaded_file)
            else:
                flowers_df = pd.read_excel(uploaded_file)

            # Validate that the required columns are present
            required_columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
            if not all(col in flowers_df.columns for col in required_columns):
                st.error("The uploaded file is missing required columns. Please check the template.")
            else:
                # Get the predictions and probabilities
                predicted_class, probabilities = predict_species(model, flowers_df)

                # Combine predictions with the original data
                probabilities_df = pd.DataFrame(probabilities, columns=[f"{classe}_probability" for classe in model.classes_])
                results_df = pd.concat([flowers_df, pd.DataFrame({'predicted Species': predicted_class}), probabilities_df], axis=1)

                # Display the results
                st.write(results_df)

                # Convert results to Excel format
                excel_data = to_excel_data(data=results_df, index=False)

                # Create a download button for the results
                st.download_button(
                    '⬇️ Download the predictions',
                    excel_data,
                    file_name='predictions.xlsx',
                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                    help='Download an Excel file with the predictions'
                )

        except Exception as e:
            st.error(f"An error occurred while processing the file: {e}")
    
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
