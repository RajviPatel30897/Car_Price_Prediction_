import pandas as pd

def load_data(file_name):
    """Load data from a CSV file."""
    try:
        data = pd.read_csv(file_name)  # Reading the file using the provided file name
        return data
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        return None

# Call the function and pass the file path as an argument
data = load_data('data/car_prices.csv')  # Pass the correct file path here
if data is not None:
    print("Data loaded successfully")
else:
    print("Failed to load data")
