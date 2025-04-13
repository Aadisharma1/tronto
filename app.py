import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Streamlit UI title and description
st.title("Crime Prediction Model")
st.write("This app predicts the crime occurrence year based on longitude and latitude.")

# File upload section for the user to upload the CSV
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    # Load the dataset
    df = pd.read_csv(uploaded_file)
    
    # Check if necessary columns exist
    if 'longitude' in df.columns and 'latitude' in df.columns and 'occurrence_year' in df.columns:
        st.write("Dataset preview:")
        st.write(df.head())  # Show the first 5 rows of the dataframe
        
        # Feature Selection
        X = df[['longitude', 'latitude']]  # Features
        y = df['occurrence_year']  # Target
        
        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Model training
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Model Evaluation
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        st.subheader("Model Performance")
        st.write(f"**Mean Squared Error (MSE):** {mse:.2f}")
        st.write(f"**R² Score:** {r2:.4f}")
        
        # Plot: Actual vs Predicted values
        st.subheader("Actual vs Predicted Plot")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(x=y_test, y=y_pred, ax=ax, alpha=0.6)
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        st.pyplot(fig)
        
        # Prediction Section
        st.subheader("Predict Crime Year")
        long = st.number_input("Longitude", value=float(X['longitude'].mean()))
        lat = st.number_input("Latitude", value=float(X['latitude'].mean()))
        
        if st.button("Predict Crime Year"):
            # Fixed this line by correcting the bracket placement
            predicted_year = model.predict([[long, lat]])[0]
            st.success(f"Predicted Occurrence Year: {predicted_year:.0f}")
    else:
        st.error("The dataset is missing required columns ('longitude', 'latitude', 'occurrence_year').")
