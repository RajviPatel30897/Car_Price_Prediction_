import streamlit as st
import pandas as pd
from src.load_data import load_data
from src.preprocess import preprocess_data
from src.train_model import train_model

# Title of the app
st.title("Car Price Prediction App")

# File uploader widget
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # Load the data from the uploaded file
    df = load_data(uploaded_file)

    if df is not None:
        # Display data preview
        st.write("Preview of the dataset:", df.head())

        # Preprocess the data
        processed_data = preprocess_data(df)

        if processed_data is not None:
            # Train the model using the processed data
            model = train_model(processed_data)

            if model is not None:
                # Take user input for prediction
                st.subheader("Enter car details for prediction:")

                # Create input fields for the user to enter car details
                brand = st.text_input("Brand")
                model_name = st.text_input("Model")
                year = st.number_input("Year", min_value=1900, max_value=2023)
                mileage = st.number_input("Mileage (in km)", min_value=0)
                fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "Electric"])

                if st.button("Predict"):
                    # Convert user input into a format the model can predict on
                    user_input = {
                        "Brand": brand,
                        "Model": model_name,
                        "Year": year,
                        "Mileage": mileage,
                        "Fuel_Type": fuel_type
                    }

                    # Preprocess user input similarly to how the data was preprocessed
                    user_input_df = pd.DataFrame([user_input])

                    # Preprocess the input to match the model's expected format
                    user_input_processed = preprocess_data(user_input_df)

                    # Align user input columns with model's training data columns
                    input_columns = processed_data.columns  # Get columns from training data
                    missing_columns = set(input_columns) - set(user_input_processed.columns)
                    for col in missing_columns:
                        user_input_processed[col] = 0  # Add missing columns with default value

                    # Ensure columns are in the same order as the training data
                    user_input_processed = user_input_processed[input_columns]

                    # Make the prediction
                    prediction = model.predict(user_input_processed)
                    st.write(f"Predicted Car Price: {prediction[0]}")
        else:
            st.error("Data preprocessing failed.")
    else:
        st.error("Error loading data.")
