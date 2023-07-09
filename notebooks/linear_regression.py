# Import Dependencies
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import time

# Load data
filepath = "../data/realtor-data.csv"

dtype_mapping = {
    'bed': float,
    'bath': float,
    'acre_lot': float,
    'house_size': float,
    'price': float
}

housing_data = pd.read_csv(filepath, dtype=dtype_mapping)

x = housing_data[['bed', 'bath', 'acre_lot', 'house_size']]
y = housing_data['price']

# Now you can check for NaN values and proceed with your analysis

# Assuming you have a DataFrame called 'data' with your input data
na_mask = housing_data.isna()  # Create a boolean mask where True indicates NaN values

# Count the number of NaN values in each column
na_counts = na_mask.sum()

# Print the columns with NaN values and their corresponding counts
print(na_counts[na_counts > 0])

# Train the model
reg = LinearRegression()
reg.fit(x, y)

 # Ask user for input
print("Want to buy a new house and want to see how prices change as the features of the house change? I just need a little more information from you to help!")

time.sleep(2)

name = input("First off, what's your name?")
print ("Okay, great to meet you " + name + " ! Let's get started!")

time.sleep(2)