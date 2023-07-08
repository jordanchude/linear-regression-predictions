# Import Dependencies
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import time

# Load data
filepath = "data/realtor-data.csv"
housing_data = pd.read_csv(filepath)

x = housing_data[['bed', 'bath', 'acre_lot', 'house_size']]
y = housing_data['price']

# Train the model
reg = LinearRegression()
reg.fit(x, y)

 # Ask user for input
print("Want to buy a new house and want to see how prices change as the features of the house change? I just need a little more information from you to help!")

time.sleep(2)

name = input("First off, what's your name?")