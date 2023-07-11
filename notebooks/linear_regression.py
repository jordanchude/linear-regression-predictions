# Import Dependencies
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import time

# Load data
filepath = "../data/realtor-data.csv"

dtype_mapping = {
    'bed': 'float',
    'bath': 'float',
    'acre_lot': 'float',
    'house_size': 'float',
    'price': 'float'
}

columns_to_include = {
    'bed': 'bed',
    'bath': 'bath',
    'acre_lot': 'acre_lot',
    'house_size': 'house_size',
    'price': 'price'
}

housing_data = pd.read_csv(filepath, dtype=dtype_mapping, usecols=list(columns_to_include.values())).dropna()

columns_to_include['bed'] = pd.to_numeric(columns_to_include['bed'], errors='coerce')
columns_to_include['bath'] = pd.to_numeric(columns_to_include['bath'], errors='coerce')
columns_to_include['acre_lot'] = pd.to_numeric(columns_to_include['acre_lot'], errors='coerce')
columns_to_include['house_size'] = pd.to_numeric(columns_to_include['house_size'], errors='coerce')
columns_to_include['price'] = pd.to_numeric(columns_to_include['price'], errors='coerce')

housing_data = pd.read_csv(filepath, dtype=dtype_mapping, usecols=columns_to_include).dropna()

print(housing_data.dtypes)

x = housing_data[['bed', 'bath', 'acre_lot', 'house_size']]
y = housing_data['price']

# Train the model
reg = LinearRegression()
reg.fit(x, y)

 # Ask user for input
print("Want to buy a new house and want to see how prices change as the features of the house change? I just need a little more information from you to help!")

time.sleep(2)
name = input("First off, what's your name? ")
print ("Okay, great to meet you " + name + " ! Let's get started!")

time.sleep(2)
bed = float(input("How many bedrooms do you want? "))
bath = float(input("How many bathrooms do you want? "))
acre_lot = float(input("How many acres do you want? "))
house_size = float(input("How big do you want your house to be (in square feet)? "))
time.sleep(2)

y_pred = reg.predict(np.array([[bed, bath, acre_lot, house_size]]))
print("Okay, based on the information you gave me, I predict that your house will cost $" + str(y_pred[0]) + " !")