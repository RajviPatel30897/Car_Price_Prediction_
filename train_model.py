import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Function to train the model
def train_model(data):
    print("Starting model training...")
    
    # Ensure 'Price' is in the dataset
    if 'Price' not in data.columns:
        raise ValueError("Dataset must contain 'Price' column for training.")
    
    # Define features (X) and target variable (y)
    X = data.drop(columns=['Price'])
    y = data['Price']
    
    # Identify categorical and numerical features
    categorical_features = X.select_dtypes(include=['object']).columns
    numerical_features = X.select_dtypes(exclude=['object']).columns
    
    # Preprocessing for categorical data
    preprocessor = ColumnTransformer([
        ('onehot', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ], remainder='passthrough')
    
    # Define model pipeline
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Save the trained model to a .pkl file
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)
    
    print("Model training completed and saved as model.pkl.")
    return model

# Run the training when the script is executed
def main():
    try:
        print("Loading dataset...")
        data = pd.read_csv("data/car_prices.csv")  # Update the path to your dataset
        print("Dataset loaded successfully.")
        
        model = train_model(data)
        print("Training completed.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()