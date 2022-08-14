import os
from time import localtime, strftime
import numpy as np
import matplotlib.pyplot as plt

from build.class_gain import GAIN
from build.utils import normalization

original_data = np.genfromtxt("datasets/mydata2/gaussian_mixture.csv", delimiter=",")
original_data = np.array(original_data, dtype="float32")

np.random.seed(666)
MISS_RATE = 0.2

myGAIN = GAIN(dim=original_data.shape[1])

LIST_EPOCHS = np.arange(start=10000, stop=100001, step=10000)
NB_REPEAT_TRAIN = 10
NB_REPEAT_IMPUTATION = 50

rmse_gain = np.zeros((NB_REPEAT_TRAIN, len(LIST_EPOCHS)))

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
        myGAIN.train(norm_data, batch_size=128, epochs=nb_epochs)
        impute_gain = np.zeros_like(miss_data)
        for i3 in range(NB_REPEAT_IMPUTATION):
            impute_gain += myGAIN.impute(norm_data)
        impute_gain /= NB_REPEAT_IMPUTATION
        rmse_gain[i1, i2] = np.sqrt(np.sum((impute_gain - norm_full_data) ** 2) / nb_miss_val)


if not os.path.exists("results/pipeline3/"):
    os.makedirs("results/pipeline3/")
if not os.path.exists("results/pipeline3/plots"):
    os.makedirs("results/pipeline3/plots")

np.save("results/pipeline3/rmse_gain.npy", rmse_gain)


# Plot: GAIN performances with many epochs
mu1 = np.mean(rmse_gain, axis=0)
std1 = np.std(rmse_gain, axis=0)
plt.figure(figsize=(6, 4))
plt.errorbar(LIST_EPOCHS, mu1, yerr=std1, color="indigo", marker=".", capsize=5.0, ls="", label="GAIN")
plt.xlabel("Nb epochs")
plt.ylabel("RMSE")
plt.title("Mixture of three Gaussian dataset [1000x5]")
plt.legend()
plt.tight_layout()
plt.savefig("results/pipeline3/plots/longRMSE.pdf")
