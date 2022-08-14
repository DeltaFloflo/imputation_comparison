import os
from time import localtime, strftime
import numpy as np
import matplotlib.pyplot as plt

from sklearn.impute import KNNImputer
from build.class_gain import GAIN
from build.class_misgan import MisGAN

from build.utils import normalization, load_dataset

np.random.seed(666)

LIST_DATASETS = ["breast", "credit", "letter", "news", "spam", "wine_red", "wine_white", "mydata1", "mydata2"]
NB_NEIGHBOURS = 50
NB_EPOCHS_GAIN = 20000
NB_EPOCHS_MISGAN = 5000
NB_REPEAT_TRAIN = 20
NB_REPEAT_IMPUTATION = 50

rmse_gain = np.zeros((NB_REPEAT_TRAIN, len(LIST_DATASETS)))
rmse_misgan = np.zeros((NB_REPEAT_TRAIN, len(LIST_DATASETS)))
rmse_knn1 = np.zeros((NB_REPEAT_TRAIN, len(LIST_DATASETS)))
rmse_knn2 = np.zeros((NB_REPEAT_TRAIN, len(LIST_DATASETS)))


for i1 in range(NB_REPEAT_TRAIN):
    print(f"\n=================", flush=True)
    print(f"==  Repeat {i1}  ==", end="       ")
    print(strftime("%Y-%m-%d %H:%M:%S", localtime()), flush=True)
    print(f"=================", flush=True)

    for i2 in range(len(LIST_DATASETS)):
        dataset_name = LIST_DATASETS[i2]
        original_data = load_dataset(dataset_name)
        original_data = np.array(original_data, dtype="float32")
        mult_fact = original_data.shape[0] / 1000.0  # Multiplicative factor for epochs and nb_neighbours
        print(f"-> Dataset = {dataset_name} {original_data.shape} | Time: ", end="", flush=True)
        print(strftime("%Y-%m-%d %H:%M:%S", localtime()), flush=True)
        r = np.random.uniform(size=original_data.shape)
        MNARprob = np.zeros(shape=original_data.shape)
        for c in range(original_data.shape[1]):
            MNARprob[:, c] = np.argsort(np.argsort(original_data[:, c])) / original_data.shape[0] * 0.90
        miss_mask = (r < MNARprob)
        nb_miss_val = np.sum(miss_mask)
        miss_data = np.copy(original_data)
        miss_data[miss_mask] = np.nan
        norm_data, norm_params = normalization(miss_data)
        norm_full_data, _ = normalization(original_data, norm_params=norm_params)

        myGAIN = GAIN(dim=original_data.shape[1])
        myMisGAN = MisGAN(dim=original_data.shape[1])
        cur_gain_epochs = int(NB_EPOCHS_GAIN / mult_fact)
        cur_misgan_epochs = int(NB_EPOCHS_MISGAN / mult_fact)
        print(f"Train GAIN for {cur_gain_epochs} epochs...", end=" ")
        myGAIN.train(norm_data, batch_size=128, epochs=cur_gain_epochs)
        print(f"Train MisGAN for {cur_misgan_epochs} epochs...", end=" ")
        myMisGAN.train(norm_data, batch_size=128, epochs=cur_misgan_epochs)
        impute_gain = np.zeros_like(miss_data)
        impute_misgan = np.zeros_like(miss_data)
        for i3 in range(NB_REPEAT_IMPUTATION):
            impute_gain += myGAIN.impute(norm_data)
            impute_misgan += myMisGAN.impute(norm_data)
        impute_gain /= NB_REPEAT_IMPUTATION
        impute_misgan /= NB_REPEAT_IMPUTATION
        rmse_gain[i1, i2] = np.sqrt(np.sum((impute_gain - norm_full_data) ** 2) / nb_miss_val)
        rmse_misgan[i1, i2] = np.sqrt(np.sum((impute_misgan - norm_full_data) ** 2) / nb_miss_val)

        print("KNN starts ", end="")  # For KNN
        cur_knn_neighbours = int(NB_NEIGHBOURS * mult_fact)
        print(f"({cur_knn_neighbours} neighbours...)")
        print(strftime("%Y-%m-%d %H:%M:%S", localtime()))
        myKNN1 = KNNImputer(n_neighbors=cur_knn_neighbours, weights="uniform", metric="nan_euclidean")
        myKNN1.fit(norm_data)
        impute_knn1 = myKNN1.transform(norm_data)
        myKNN2 = KNNImputer(n_neighbors=cur_knn_neighbours, weights="distance", metric="nan_euclidean")
        myKNN2.fit(norm_data)
        impute_knn2 = myKNN2.transform(norm_data)
        rmse_knn1[i1, i2] = np.sqrt(np.sum((impute_knn1 - norm_full_data) ** 2) / nb_miss_val)
        rmse_knn2[i1, i2] = np.sqrt(np.sum((impute_knn2 - norm_full_data) ** 2) / nb_miss_val)
        print("")


if not os.path.exists("results/pipeline11/"):
    os.makedirs("results/pipeline11/")
if not os.path.exists("results/pipeline11/plots"):
    os.makedirs("results/pipeline11/plots")


np.save("results/pipeline11/rmse_gain.npy", rmse_gain)
np.save("results/pipeline11/rmse_misgan.npy", rmse_misgan)
np.save("results/pipeline11/rmse_knn1.npy", rmse_knn1)
np.save("results/pipeline11/rmse_knn2.npy", rmse_knn2)


# Plot 1: Complete performances for everyone
mu1 = np.mean(rmse_gain, axis=0)
std1 = np.std(rmse_gain, axis=0)
mu2 = np.mean(rmse_misgan, axis=0)
std2 = np.std(rmse_misgan, axis=0)
mu3 = np.mean(rmse_knn1, axis=0)
std3 = np.std(rmse_knn1, axis=0)
mu4 = np.mean(rmse_knn2, axis=0)
std4 = np.std(rmse_knn2, axis=0)
x_ticks_locs = np.arange(0.0, 9.0)
plt.figure(figsize=(8, 4))
plt.errorbar(x_ticks_locs-0.15, mu1, yerr=std1, color="indigo", marker=".", capsize=5.0, ls="", label="GAIN")
plt.errorbar(x_ticks_locs-0.05, mu2, yerr=std2, color="grey", marker=".", capsize=5.0, ls="", label="MisGAN")
plt.errorbar(x_ticks_locs+0.05, mu3, yerr=std3, color="teal", marker=".", capsize=5.0, ls="", label="KNN-uniform")
plt.errorbar(x_ticks_locs+0.15, mu4, yerr=std4, color="peru", marker=".", capsize=5.0, ls="", label="KNN-distance")
plt.xticks(x_ticks_locs, LIST_DATASETS, rotation ="vertical")
plt.ylabel("RMSE")
plt.title("[MNAR setting] mean missing_rate of 45%")
plt.legend()
plt.tight_layout()
plt.savefig("results/pipeline11/plots/rmse.pdf")


# Plot 2: Remove the bad performances for the dataset News
mu1 = np.mean(rmse_gain, axis=0)[[0, 1, 2, 4, 5, 6, 7, 8]]
std1 = np.std(rmse_gain, axis=0)[[0, 1, 2, 4, 5, 6, 7, 8]]
mu2 = np.mean(rmse_misgan, axis=0)[[0, 1, 2, 4, 5, 6, 7, 8]]
std2 = np.std(rmse_misgan, axis=0)[[0, 1, 2, 4, 5, 6, 7, 8]]
mu3 = np.mean(rmse_knn1, axis=0)[[0, 1, 2, 4, 5, 6, 7, 8]]
std3 = np.std(rmse_knn1, axis=0)[[0, 1, 2, 4, 5, 6, 7, 8]]
mu4 = np.mean(rmse_knn2, axis=0)[[0, 1, 2, 4, 5, 6, 7, 8]]
std4 = np.std(rmse_knn2, axis=0)[[0, 1, 2, 4, 5, 6, 7, 8]]
x_ticks_locs = np.arange(0.0, 8.0)
x_ticks_names = ["breast", "credit", "letter", "spam", "wine_red", "wine_white", "mydata1", "mydata2"]
plt.figure(figsize=(8, 4))
plt.errorbar(x_ticks_locs-0.15, mu1, yerr=std1, color="indigo", marker=".", capsize=5.0, ls="", label="GAIN")
plt.errorbar(x_ticks_locs-0.05, mu2, yerr=std2, color="grey", marker=".", capsize=5.0, ls="", label="MisGAN")
plt.errorbar(x_ticks_locs+0.05, mu3, yerr=std3, color="teal", marker=".", capsize=5.0, ls="", label="KNN-uniform")
plt.errorbar(x_ticks_locs+0.15, mu4, yerr=std4, color="peru", marker=".", capsize=5.0, ls="", label="KNN-distance")
plt.xticks(x_ticks_locs, x_ticks_names, rotation ="vertical")
plt.ylabel("RMSE")
plt.title("[MNAR setting] mean missing_rate of 45%")
plt.legend()
plt.tight_layout()
plt.savefig("results/pipeline11/plots/rmse2.pdf")


# Plot 2: Remove the bad performances for the dataset News
mu1 = np.mean(rmse_gain, axis=0)[[0, 1, 2, 4, 5, 6, 7, 8]]
std1 = np.std(rmse_gain, axis=0)[[0, 1, 2, 4, 5, 6, 7, 8]]
mu3 = np.mean(rmse_knn1, axis=0)[[0, 1, 2, 4, 5, 6, 7, 8]]
std3 = np.std(rmse_knn1, axis=0)[[0, 1, 2, 4, 5, 6, 7, 8]]
mu4 = np.mean(rmse_knn2, axis=0)[[0, 1, 2, 4, 5, 6, 7, 8]]
std4 = np.std(rmse_knn2, axis=0)[[0, 1, 2, 4, 5, 6, 7, 8]]
x_ticks_locs = np.arange(0.0, 8.0)
x_ticks_names = ["breast", "credit", "letter", "spam", "wine_red", "wine_white", "mydata1", "mydata2"]
plt.figure(figsize=(8, 4))
plt.errorbar(x_ticks_locs-0.15, mu1, yerr=std1, color="indigo", marker=".", capsize=5.0, ls="", label="GAIN")
plt.errorbar(x_ticks_locs, mu3, yerr=std3, color="teal", marker=".", capsize=5.0, ls="", label="KNN-uniform")
plt.errorbar(x_ticks_locs+0.15, mu4, yerr=std4, color="peru", marker=".", capsize=5.0, ls="", label="KNN-distance")
plt.xticks(x_ticks_locs, x_ticks_names, rotation ="vertical")
plt.ylabel("RMSE")
plt.title("[MNAR setting] mean missing_rate of 45%")
plt.legend()
plt.tight_layout()
plt.savefig("results/pipeline11/plots/rmse3.pdf")