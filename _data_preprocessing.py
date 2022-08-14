import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# For breast
data = np.genfromtxt("datasets/breast/wdbc.data", delimiter=",")
data = data[:, 2:] # We only keep the last 30 columns
print(np.any(np.isnan(data))) # There is no NaN at all!
print(data.shape) # 569 observations and 30 continuous variables (OK!)
names = open("datasets/breast/wdbc.names")
print(names.read())

# For spam
data = np.genfromtxt("datasets/spam/spambase.data", delimiter=",")
data = data[:, :-1] # Get rid of the last column (the label: spam or not)
print(np.any(np.isnan(data))) # There is no NaN!
print(data.shape) # 4601 observations and 57 continuous variables (OK!)
names = open("datasets/spam/spambase.names")
print(names.read())

# For letter
data = np.genfromtxt("datasets/letter/letter-recognition.data", delimiter=",")
data = data[:, 1:] # Get rid of the first column (the actual letter)
print(np.any(np.isnan(data))) # There is no NaN!
print(data.shape) # 20000 observations and 16 cat. var. (int between 0 and 15)
data_flat = data.flatten()
plt.figure()
plt.hist(data_flat, bins=100)
plt.show()

# For credit
data = pd.read_excel("datasets/credit/default_clients.xls")
data = np.array(data.iloc[1:, 1:-1], dtype="float32") # Get rid of the labels and the irrelevant columns
print(np.any(np.isnan(data))) # There is no NaN!
data = np.delete(data, [1, 2, 3], axis=1)  # Remove the socio-demographics categorical variables
data = np.delete(data, [2, 3, 4, 5, 6, 7], axis=1)  # Remove the past payment records (categorical)
print(data.shape) # 30000 observations and 14 cont. + 9 cat. var. (total 23)

# For news
data = np.genfromtxt("datasets/news/OnlineNewsPopularity.csv", delimiter=",")
data = data[1:, 2:-1] # Remove labels and irrelevant columns
print(np.any(np.isnan(data))) # There is no NaN!
data = np.delete(data, [11, 12, 13, 14, 15, 16], axis=1)  # Remove data channel (binary)
data = np.delete(data, [23, 24, 25, 26, 27, 28, 29, 30], axis=1)  # Remove what day of the week (binary)
print(data.shape) # 39644 observations (AND NOT 39797!!!) and 44 cont.

# For wine (RED)
data = np.genfromtxt("datasets/wine/winequality-red.csv", delimiter=";")
data = data[1:, :] # Remove labels
print(np.any(np.isnan(data))) # There is no NaN!
print(data.shape) # 1599 obs. and 11 continuous variables + last one (score) int between 0 and 10

# For wine (WHITE)
data = np.genfromtxt("datasets/wine/winequality-white.csv", delimiter=";")
data = data[1:, :] # Remove labels
print(np.any(np.isnan(data))) # There is no NaN!
print(data.shape) # 4898 obs. and 11 continuous variables + last one (score) int between 0 and 10

# My own dataset 1 (One multivariate gaussian)
data = np.genfromtxt("datasets/mydata1/multivariate_gauss.csv", delimiter=",")
print(np.any(np.isnan(data))) # There is no NaN!
print(data.shape) # 1000 obs. and 5 continuous variables

# My own dataset 2
data = np.genfromtxt("datasets/mydata2/gaussian_mixture.csv", delimiter=",")
print(np.any(np.isnan(data))) # There is no NaN!
print(data.shape) # 1000 obs. and 5 continuous variables