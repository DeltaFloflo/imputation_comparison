import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


### ======================== ###
#  Multivariate Gaussian data  #
### ======================== ###

seed = 31415
np.random.seed(seed) # For reproducibility

dim = 5 # dimension of the matrix
k = 2 # number of factors
A = np.random.normal(size=(k, dim))
C = np.diag(np.random.uniform(low=0.0, high=1.0, size=dim)) # bonus diagonal matrix
W = np.transpose(A) @ A + C
Sigma = np.diag(np.sqrt(1.0 / np.diag(W))) @ W @ np.diag(np.sqrt(1.0 / np.diag(W)))
Mu = np.random.normal(loc=0.0, scale=2.0, size=dim)

N = 1000
data = np.random.multivariate_normal(mean=Mu, cov=Sigma, size=N)
pp_data = pd.DataFrame(data)

plt.figure(figsize=(10, 10))  # Plot generated data
pp = sns.PairGrid(pp_data)
pp.map_diag(sns.histplot, kde=True, stat="density", common_norm=False)
pp.map_offdiag(sns.scatterplot, marker="+", linewidth=1)
plt.savefig("datasets/mydata1/pairplot.pdf")
plt.close("all")

np.savetxt("datasets/mydata1/multivariate_gauss_check.csv", data, delimiter=",")


### ====================== ###
#  Mixture of Gaussian data  #
### ====================== ###

seed = 31415
np.random.seed(seed) # For reproducibility

dim = 5 # dimension of the matrix
k = 2 # number of factors
N_size = [150, 300, 550] # Proportion 15%, 30%, 55% (total 1000 data points)

for n in range(3):
    A = np.random.normal(size=(k, dim))
    C = np.diag(np.random.uniform(low=0.0, high=1.0, size=dim)) # bonus diagonal matrix
    W = np.transpose(A) @ A + C
    Sigma = np.diag(np.sqrt(1.0 / np.diag(W))) @ W @ np.diag(np.sqrt(1.0 / np.diag(W)))
    Mu = np.random.normal(loc=0.0, scale=4.0, size=dim)
    sample = np.random.multivariate_normal(mean=Mu, cov=Sigma, size=N_size[n])
    if n==0:
        data = sample
    else:
        data = np.concatenate((data, sample))

pp_data = pd.DataFrame(data)
pp_data["label"] = ["Class0"] * N_size[0] + ["Class1"] * N_size[1] + ["Class2"] * N_size[2]
plt.figure(figsize=(10, 10))  # Plot generated data
pp = sns.PairGrid(pp_data, hue="label")
pp.map_diag(sns.histplot, kde=True, stat="density", common_norm=False)
pp.map_offdiag(sns.scatterplot, marker="+", linewidth=1)
pp.add_legend(title="Label", adjust_subtitles=True)
pp.savefig("datasets/mydata2/pairplot.pdf")
plt.close("all")

np.savetxt("datasets/mydata2/gaussian_mixture.csv", data, delimiter=",")