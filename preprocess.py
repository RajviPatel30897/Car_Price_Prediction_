import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_data(df):
    """Preprocess the car price dataset."""
    
    # Handle missing values for numeric columns
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    
    # Handle missing values for non-numeric columns (e.g., categorical)
    non_numeric_cols = df.select_dtypes(exclude=['float64', 'int64']).columns
    for col in non_numeric_cols:
        # You can choose to fill with the most frequent value or drop rows
        df[col] = df[col].fillna(df[col].mode()[0])
    
    # Convert categorical variables to numeric using Label Encoding
    label_encoder = LabelEncoder()
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = label_encoder.fit_transform(df[col])

    return df
