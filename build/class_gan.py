import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.models import load_model

from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestRegressor

from build.utils import reset_weights


def make_GANgen():
    model = Sequential()
    model.add(Dense(128, activation="relu", input_shape=(6,)))
    model.add(BatchNormalization())
    model.add(Dense(128, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dense(128, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dense(6, activation="sigmoid"))
    return model


def make_GANdisc():
    model = Sequential()
    model.add(Dense(128, activation="relu", input_shape=(6,)))
    model.add(Dropout(rate=0.3))
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(rate=0.3))
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(rate=0.3))
    model.add(Dense(1, activation="sigmoid"))
    return model


cross_entropy = tf.keras.losses.BinaryCrossentropy()


class GAN():

    def __init__(self):
        self.G = make_GANgen()
        self.D = make_GANdisc()
        self.Goptim = tf.keras.optimizers.Adam(1e-4)
        self.Doptim = tf.keras.optimizers.Adam(1e-4)
        self.trained = False
        self.nb_epochs = 0
        self.Gloss = []
        self.Dloss = []


    @staticmethod
    def compute_D_loss(real_output, fake_output):
        real_loss = cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss


    @staticmethod
    def compute_G_loss(fake_output):
        return cross_entropy(tf.ones_like(fake_output), fake_output)


    def reinitialize(self):
        reset_weights(self.G)
        reset_weights(self.D)
        self.trained = False
        self.nb_epochs = 0
        self.Gloss = []
        self.Dloss = []


    @tf.function  # Causes the function to be "compiled"
    def train_step(self, batch_planets):
        cur_batch_size = batch_planets.shape[0]
        noise = tf.random.normal([cur_batch_size, 6])

        with tf.GradientTape() as G_tape, tf.GradientTape() as D_tape:
            fake_planets = self.G(noise, training=True) # Create fake planets
            D_real = self.D(batch_planets, training=True) # Predictions of D for the real planets (D wants 1)
            D_fake = self.D(fake_planets, training=True) # Predictions of D for the fake planets (D wants 0, G wants 1)

            G_loss = self.compute_G_loss(D_fake)
            D_loss = self.compute_D_loss(D_real, D_fake)

            Ggrad = G_tape.gradient(G_loss, self.G.trainable_variables)
            Dgrad = D_tape.gradient(D_loss, self.D.trainable_variables)

            self.Goptim.apply_gradients(zip(Ggrad, self.G.trainable_variables))
            self.Doptim.apply_gradients(zip(Dgrad, self.D.trainable_variables))

            return G_loss, D_loss


    def train(self, dataset, batch_size, epochs):
        nb_lines = dataset.shape[0]
        nb_batch = int(np.ceil(nb_lines / batch_size))
        list_indices = np.arange(nb_lines)
        G_loss_store = np.zeros(epochs, dtype="float32")
        D_loss_store = np.zeros(epochs, dtype="float32")
        for epoch in range(epochs):
            if epoch % 100 == 0:
                print(epoch, end=" ", flush=True)
                if (epoch > 0) and (epoch % 500 == 0):
                    print("")
            G_temp = []
            D_temp = []
            np.random.shuffle(list_indices)
            for i in range(nb_batch):
                idx = list_indices[(i * batch_size):((i + 1) * batch_size)]
                cbs = idx.shape[0]  # cbs = current batch size
                planet_batch = dataset[idx]
                G_loss, D_loss = self.train_step(planet_batch)
                G_temp.append(G_loss.numpy() * cbs)
                D_temp.append(D_loss.numpy() * cbs)
            G_loss_store[epoch] = np.array(G_temp).mean() / nb_lines
            D_loss_store[epoch] = np.array(D_temp).mean() / nb_lines
        print("")
        self.trained = True
        self.nb_epochs = epochs
        self.Gloss = G_loss_store
        self.Dloss = D_loss_store


    def save_weights(self, out_dir):
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        self.G.save(out_dir + "G.h5")
        self.D.save(out_dir + "D.h5")


    def plot_distrib(self, out_dir, fake_data_size, train_planets):
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        nb_train_planets = train_planets.shape[0]
        noise = tf.random.normal([fake_data_size, 6])  # Generate 1000 fake planets and plot them
        fake_planets = self.G(noise, training=False).numpy()
        pp_data = np.concatenate((train_planets, fake_planets))  # pairplot data
        pp_data = pd.DataFrame(pp_data)
        pp_data = pp_data.rename(
            {0: "radius", 1: "mass", 2: "period", 3: "temp", 4: "nb planets", 5: "star mass"}, axis="columns")
        pp_data["label"] = ["true"] * nb_train_planets + ["fake"] * fake_data_size

        plt.figure(figsize=(10, 10))  # Plot generated planets
        pp = sns.PairGrid(pp_data, hue="label")
        pp.map_diag(sns.histplot, kde=True, stat="density", common_norm=False)
        pp.map_offdiag(sns.scatterplot, marker="+", linewidth=1)
        pp.add_legend(title="Label", adjust_subtitles=True)
        pp.savefig(out_dir + "pp.pdf")
        plt.close("all")


    def load_params(self, path):
        G_path = path + "G.h5"
        D_path = path + "D.h5"
        self.G = load_model(G_path, compile=False)
        self.D = load_model(D_path, compile=False)


    def transit_imputeKNN(self, nandata, fake_data_size, nb_neighbours):
        noise = tf.random.normal([fake_data_size, 6])
        fake_planets = self.G(noise, training=False).numpy()

        knn_planets_train = fake_planets[:, [0, 2, 3, 4, 5]]  # Generated planets to train
        knn_planets_impute = nandata[:, [0, 2, 3, 4, 5]]
        KNN = NearestNeighbors(n_neighbors=nb_neighbours)
        KNN.fit(knn_planets_train)
        knn_out = KNN.kneighbors(knn_planets_impute, n_neighbors=nb_neighbours)  # Find the nearest neighbours

        imputed_masses = np.zeros(nandata.shape[0])
        for i in range(nandata.shape[0]):
            list_idx = knn_out[1][i]
            knn_masses = fake_planets[list_idx][:, 1]
            imputed_masses[i] = knn_masses.mean(axis=0)  # Prediction by the mean
        return imputed_masses


    def transit_imputeRF(self, nandata, fake_data_size, nb_trees):
        noise = tf.random.normal([fake_data_size, 6])
        fake_planets = self.G(noise, training=False).numpy()

        rf_planets_x = fake_planets[:, [0, 2, 3, 4, 5]]  # Training columns for the RF
        rf_planets_y = fake_planets[:, 1].ravel()  # Target columns for the RF
        rf_planets_impute = nandata[:, [0, 2, 3, 4, 5]]
        RF = RandomForestRegressor(n_estimators=nb_trees)
        RF.fit(rf_planets_x, rf_planets_y)  # Fit model from X to y
        imputed_masses = RF.predict(rf_planets_impute)  # Prediction by the RF
        return imputed_masses


    def rv_imputeKNN(self, nandata, fake_data_size, nb_neighbours):
        noise = tf.random.normal([fake_data_size, 6])
        fake_planets = self.G(noise, training=False).numpy()

        knn_planets_train = fake_planets[:, [2, 3, 4, 5]]  # Generated planets to train
        knn_planets_impute = nandata[:, [2, 3, 4, 5]]
        KNN = NearestNeighbors(n_neighbors=nb_neighbours)
        KNN.fit(knn_planets_train)
        knn_out = KNN.kneighbors(knn_planets_impute, n_neighbors=nb_neighbours)  # Find the nearest neighbours

        imputed_radmass = np.zeros(nandata.shape[0]) # TO DO: DO SOMETHING HERE!!
        for i in range(nandata.shape[0]):
            list_idx = knn_out[1][i]
            knn_masses = fake_planets[list_idx][:, 1]
            imputed_radmass[i] = knn_masses.mean(axis=0)  # Prediction by the mean
        return imputed_radmass


    def rv_imputeRF(self, nandata, fake_data_size, nb_trees):
        noise = tf.random.normal([fake_data_size, 6])
        fake_planets = self.G(noise, training=False).numpy()

