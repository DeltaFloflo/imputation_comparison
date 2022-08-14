import os
from time import localtime, strftime
import numpy as np
import matplotlib.pyplot as plt

from sklearn.impute import KNNImputer
from build.class_gain import GAIN
from build.class_misgan import MisGAN

from build.utils import normalization

np.random.seed(667)

NB_NEIGHBOURS = 50
NB_EPOCHS_GAIN = 20000
NB_EPOCHS_MISGAN = 5000
NB_REPEAT_TRAIN = 20
NB_REPEAT_IMPUTATION = 50

rmse_gain = np.zeros((NB_REPEAT_TRAIN, 2))
rmse_misgan = np.zeros((NB_REPEAT_TRAIN, 2))
rmse_knn1 = np.zeros((NB_REPEAT_TRAIN, 2))
rmse_knn2 = np.zeros((NB_REPEAT_TRAIN, 2))

for i1 in range(NB_REPEAT_TRAIN):
    print(f"\n== Repeat {i1} ==", flush=True)
    print(strftime("%Y-%m-%d %H:%M:%S", localtime()), flush=True)

    for i2 in range(2):
        if i2==0:
            original_data = np.genfromtxt("datasets/mydata1/multivariate_gauss.csv", delimiter=",")
            original_data = np.array(original_data, dtype="float32")
            print("Univariate Gaussian dataset...")
        if i2==1:
            original_data = np.genfromtxt("datasets/mydata2/gaussian_mixture.csv", delimiter=",")
            original_data = np.array(original_data, dtype="float32")
            print("Mixture of Gaussian dataset...")
        myGAIN = GAIN(dim=original_data.shape[1])
        myMisGAN = MisGAN(dim=original_data.shape[1])

        r = np.random.uniform(size=original_data.shape)
        miss_mask = r > 2.0  # Create miss_mass with False everywehre
        m1 = np.min(original_data[:, 0])
        m2 = np.max(original_data[:, 0])
        probs = (original_data[:, 0] - m1) / (m2 - m1 + 1e-6)
        for c in range(1, original_data.shape[1]):
            miss_mask[:, c] = (r[:, c] < probs)  # An example of MAR scenario
        nb_miss_val = np.sum(miss_mask)
        miss_data = np.copy(original_data)
        miss_data[miss_mask] = np.nan
        norm_data, norm_params = normalization(miss_data)
        norm_full_data, _ = normalization(original_data, norm_params=norm_params)

        myGAIN.reinitialize()  # For GAIN and MisGAN
        myMisGAN.reinitialize()
        myGAIN.train(norm_data, batch_size=128, epochs=NB_EPOCHS_GAIN)
        myMisGAN.train(norm_data, batch_size=128, epochs=NB_EPOCHS_MISGAN)
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
        myKNN1 = KNNImputer(n_neighbors=NB_NEIGHBOURS, weights="uniform", metric="nan_euclidean")
        myKNN1.fit(norm_data)
        impute_knn1 = myKNN1.transform(norm_data)
        myKNN2 = KNNImputer(n_neighbors=NB_NEIGHBOURS, weights="distance", metric="nan_euclidean")
        myKNN2.fit(norm_data)
        impute_knn2 = myKNN2.transform(norm_data)
        rmse_knn1[i1, i2] = np.sqrt(np.sum((impute_knn1 - norm_full_data) ** 2) / nb_miss_val)
        rmse_knn2[i1, i2] = np.sqrt(np.sum((impute_knn2 - norm_full_data) ** 2) / nb_miss_val)


if not os.path.exists("results/pipeline7/"):
    os.makedirs("results/pipeline7/")
if not os.path.exists("results/pipeline7/plots"):
    os.makedirs("results/pipeline7/plots")


np.save("results/pipeline7/rmse_gain.npy", rmse_gain)
np.save("results/pipeline7/rmse_misgan.npy", rmse_misgan)
np.save("results/pipeline7/rmse_knn1.npy", rmse_knn1)
np.save("results/pipeline7/rmse_knn2.npy", rmse_knn2)


# Plot 1: Complete performances for everyone
mu1 = np.mean(rmse_gain, axis=0)
std1 = np.std(rmse_gain, axis=0)
mu2 = np.mean(rmse_misgan, axis=0)
std2 = np.std(rmse_misgan, axis=0)
mu3 = np.mean(rmse_knn1, axis=0)
std3 = np.std(rmse_knn1, axis=0)
mu4 = np.mean(rmse_knn2, axis=0)
std4 = np.std(rmse_knn2, axis=0)
x_ticks_locs = np.arange(0.0, 2.0)
x_ticks_names = ["One Gauss.", "Mixture"]
plt.figure(figsize=(8, 4))
plt.errorbar(x_ticks_locs-0.15, mu1, yerr=std1, color="indigo", marker=".", capsize=5.0, ls="", label="GAIN")
plt.errorbar(x_ticks_locs-0.05, mu2, yerr=std2, color="grey", marker=".", capsize=5.0, ls="", label="MisGAN")
plt.errorbar(x_ticks_locs+0.05, mu3, yerr=std3, color="teal", marker=".", capsize=5.0, ls="", label="KNN-uniform")
plt.errorbar(x_ticks_locs+0.15, mu4, yerr=std4, color="peru", marker=".", capsize=5.0, ls="", label="KNN-distance")
plt.xticks(x_ticks_locs, x_ticks_names)
plt.xlabel("Gaussian datasets [1000x5]")
plt.ylabel("RMSE")
plt.title("MAR scenario (based on first column)")
plt.legend()
plt.tight_layout()
plt.savefig("results/pipeline7/plots/rmse.pdf")

