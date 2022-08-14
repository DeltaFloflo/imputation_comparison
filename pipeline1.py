import os
from time import localtime, strftime
import numpy as np
import matplotlib.pyplot as plt

from sklearn.impute import KNNImputer
from build.class_gain import GAIN
from build.class_misgan import MisGAN

from build.utils import normalization

original_data = np.genfromtxt("datasets/mydata1/multivariate_gauss.csv", delimiter=",")
original_data = np.array(original_data, dtype="float32")

np.random.seed(666)
MISS_RATE = 0.2

myGAIN = GAIN(dim=original_data.shape[1])
myMisGAN = MisGAN(dim=original_data.shape[1])

LIST_EPOCHS = np.arange(start=1000, stop=20001, step=1000)
LIST_NEIGHBOURS = np.arange(start=2, stop=301, step=2)
NB_REPEAT_TRAIN = 20
NB_REPEAT_IMPUTATION = 50

rmse_gain = np.zeros((NB_REPEAT_TRAIN, len(LIST_EPOCHS)))
rmse_misgan = np.zeros((NB_REPEAT_TRAIN, len(LIST_EPOCHS)))
rmse_knn1 = np.zeros((NB_REPEAT_TRAIN, len(LIST_NEIGHBOURS)))
rmse_knn2 = np.zeros((NB_REPEAT_TRAIN, len(LIST_NEIGHBOURS)))

for i1 in range(NB_REPEAT_TRAIN):
    print(f"\n== Repeat {i1} ==", flush=True)
    r = np.random.uniform(size=original_data.shape)
    miss_mask = (r < MISS_RATE)
    nb_miss_val = np.sum(miss_mask)
    miss_data = np.copy(original_data)
    miss_data[miss_mask] = np.nan
    norm_data, norm_params = normalization(miss_data)
    norm_full_data, _ = normalization(original_data, norm_params=norm_params)

    for i2 in range(len(LIST_EPOCHS)): # For GAIN and MisGAN
        nb_epochs = LIST_EPOCHS[i2]
        print(f"-> Nb epochs = {nb_epochs} | Time: ", end="", flush=True)
        print(strftime("%Y-%m-%d %H:%M:%S", localtime()), flush=True)
        myGAIN.reinitialize()
        myMisGAN.reinitialize()
        myGAIN.train(norm_data, batch_size=128, epochs=nb_epochs)
        myMisGAN.train(norm_data, batch_size=128, epochs=nb_epochs)
        impute_gain = np.zeros_like(miss_data)
        impute_misgan = np.zeros_like(miss_data)
        for i3 in range(NB_REPEAT_IMPUTATION):
            impute_gain += myGAIN.impute(norm_data)
            impute_misgan += myMisGAN.impute(norm_data)
        impute_gain /= NB_REPEAT_IMPUTATION
        impute_misgan /= NB_REPEAT_IMPUTATION
        rmse_gain[i1, i2] = np.sqrt(np.sum((impute_gain - norm_full_data) ** 2) / nb_miss_val)
        rmse_misgan[i1, i2] = np.sqrt(np.sum((impute_misgan - norm_full_data) ** 2) / nb_miss_val)

    print("KNN starts... ", end="")
    print(strftime("%Y-%m-%d %H:%M:%S", localtime()))
    for i2 in range(len(LIST_NEIGHBOURS)): # For KNN
        nb_neighbours = LIST_NEIGHBOURS[i2]
        myKNN1 = KNNImputer(n_neighbors=nb_neighbours, weights="uniform", metric="nan_euclidean")
        myKNN1.fit(norm_data)
        impute_knn1 = myKNN1.transform(norm_data)
        myKNN2 = KNNImputer(n_neighbors=nb_neighbours, weights="distance", metric="nan_euclidean")
        myKNN2.fit(norm_data)
        impute_knn2 = myKNN2.transform(norm_data)
        rmse_knn1[i1, i2] = np.sqrt(np.sum((impute_knn1 - norm_full_data) ** 2) / nb_miss_val)
        rmse_knn2[i1, i2] = np.sqrt(np.sum((impute_knn2 - norm_full_data) ** 2) / nb_miss_val)


if not os.path.exists("results/pipeline1/"):
    os.makedirs("results/pipeline1/")
if not os.path.exists("results/pipeline1/plots"):
    os.makedirs("results/pipeline1/plots")

np.save("results/pipeline1/rmse_gain.npy", rmse_gain)
np.save("results/pipeline1/rmse_misgan.npy", rmse_misgan)
np.save("results/pipeline1/rmse_knn1.npy", rmse_knn1)
np.save("results/pipeline1/rmse_knn2.npy", rmse_knn2)


# Plot 1: GAIN and MisGAN performances
mu1 = np.mean(rmse_gain, axis=0)
std1 = np.std(rmse_gain, axis=0)
mu2 = np.mean(rmse_misgan, axis=0)
std2 = np.std(rmse_misgan, axis=0)
plt.figure(figsize=(6, 4))
plt.errorbar(LIST_EPOCHS-15, mu1, yerr=std1, color="indigo", marker=".", capsize=5.0, ls="", label="GAIN")
plt.errorbar(LIST_EPOCHS+15, mu2, yerr=std2, color="grey", marker=".", capsize=5.0, ls="", label="MisGAN")
plt.xlabel("Nb epochs")
plt.ylabel("RMSE")
plt.title("Multivariate Gaussian dataset [1000x5]")
plt.legend()
plt.tight_layout()
plt.savefig("results/pipeline1/plots/rmse1.pdf")


# Plot 2: KNN performances
mu1 = np.mean(rmse_knn1, axis=0)
std1 = np.std(rmse_knn1, axis=0)
mu2 = np.mean(rmse_knn2, axis=0)
std2 = np.std(rmse_knn2, axis=0)
plt.figure(figsize=(6, 4))
plt.plot(LIST_NEIGHBOURS, mu1, color="indigo", label="KNN-uniform")
plt.fill_between(LIST_NEIGHBOURS, mu1-std1, mu1+std1, color="indigo", alpha=0.3)
plt.plot(LIST_NEIGHBOURS, mu2, color="grey", label="KNN-distance")
plt.fill_between(LIST_NEIGHBOURS, mu2-std2, mu2+std2, color="grey", alpha=0.3)
plt.xlabel("Nb neighbours")
plt.xlim(left=0.0)
plt.ylabel("RMSE")
plt.title("Multivariate Gaussian dataset [1000x5]")
plt.legend()
plt.tight_layout()
plt.savefig("results/pipeline1/plots/rmse2.pdf")

