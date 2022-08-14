import os

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout

from build.utils import reset_weights
from build.utils import drawHintMatrix


def make_GAINgen(dim):
    model = Sequential()
    model.add(Dense(64, activation="relu", input_shape=(2*dim,))) # Input shape = 12 for GAIN
    model.add(BatchNormalization()) # Therefore 32 outputs for the first layer (to have about same number of params)
    model.add(Dense(128, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dense(128, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dense(dim, activation="sigmoid"))
    return model


def make_GAINdisc(dim):
    model = Sequential()
    model.add(Dense(64, activation="relu", input_shape=(2*dim,))) # Input shape = 12 for GAIN
    model.add(Dropout(rate=0.3)) # Therefore 32 outputs for the first layer (to have about same number of params)
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(rate=0.3))
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(rate=0.3))
    model.add(Dense(dim, activation="sigmoid")) # In GAIN, the discriminator outputs a 6-D vector
    return model


class GAIN():

    def __init__(self, dim):
        self.dim = dim
        self.G = make_GAINgen(dim)
        self.D = make_GAINdisc(dim)
        self.Goptim = tf.keras.optimizers.Adam(1e-4)
        self.Doptim = tf.keras.optimizers.Adam(1e-4)
        self.alpha = 100.0
        self.hint_rate = 0.9
        self.trained = False
        self.nb_epochs = 0
        self.Gloss1 = []
        self.Gloss2 = []
        self.Dloss = []


    @staticmethod
    def compute_D_loss(D_output, M, H):
        L1 = M * tf.math.log(D_output + 1e-6)
        L2 = (1.0 - M) * tf.math.log(1.0 - D_output + 1e-6)
        L = - (L1 + L2) * tf.cast((H == 0.5), dtype=tf.float32)
        nb_cells = tf.math.reduce_sum(tf.cast((H == 0.5), dtype=tf.float32))
        if nb_cells == 0:
            loss = 0.0
        else:
            loss = tf.math.reduce_sum(L) / nb_cells
        return loss


    @staticmethod
    def compute_G_loss(G_output, D_output, X, M, H):
        Ltemp = - ((1.0 - M) * tf.math.log(D_output + 1e-6))
        L = Ltemp * tf.cast((H == 0.5), dtype=tf.float32)
        nb_cells1 = tf.math.reduce_sum(tf.cast((H == 0.5), dtype=tf.float32))
        if nb_cells1 == 0:
            loss1 = 0.0
        else:
            loss1 = tf.math.reduce_sum(L) / nb_cells1  # Loss for G to fool D
        squared_err = ((X - G_output) ** 2) * M
        nb_cells2 = tf.math.reduce_sum(M)
        if nb_cells2 == 0:
            loss2 = 0.0
        else:
            loss2 = tf.math.reduce_sum(squared_err) / nb_cells2
        return loss1, loss2  # loss1 is cross-entropy, loss2 is RMSE


    def reinitialize(self):
        reset_weights(self.G)
        reset_weights(self.D)
        self.trained = False
        self.nb_epochs = 0
        self.Gloss1 = []
        self.Gloss2 = []
        self.Dloss = []


    @tf.function  # Causes the function to be "compiled"
    def train_step(self, batch_planets):
        cur_batch_size = batch_planets.shape[0]
        noise = tf.random.normal([cur_batch_size, self.dim], dtype=tf.float32)
        M = 1.0 - tf.cast(tf.math.is_nan(batch_planets), dtype=tf.float32)  # 0=NaN, 1=obs.
        X = tf.where(tf.math.is_nan(batch_planets), noise, batch_planets)
        G_input = tf.concat((X, M), axis=1)

        with tf.GradientTape() as G_tape, tf.GradientTape() as D_tape:
            G_output = self.G(G_input, training=True)  # G outputs 6 dim fake observations
            X_hat = X * M + G_output * (1.0 - M)
            Htemp = tf.cast(drawHintMatrix(self.hint_rate, cur_batch_size, self.dim), dtype=tf.float32)
            H = M * Htemp + 0.5 * (1.0 - Htemp)
            D_input = tf.concat((X_hat, H), axis=1)
            D_output = self.D(D_input, training=True)

            D_loss = self.compute_D_loss(D_output, M, H)  # The GAIN losses are different for D and G
            G_loss1, G_loss2 = self.compute_G_loss(G_output, D_output, X, M, H)  # Check their definitions above...
            G_loss = G_loss1 + self.alpha * G_loss2  # Use the hyperparam alpha here

            G_gradients = G_tape.gradient(G_loss, self.G.trainable_variables)
            D_gradients = D_tape.gradient(D_loss, self.D.trainable_variables)

            self.Goptim.apply_gradients(zip(G_gradients, self.G.trainable_variables))
            self.Doptim.apply_gradients(zip(D_gradients, self.D.trainable_variables))

            return G_loss1, G_loss2, D_loss


    def train(self, dataset, batch_size, epochs):
        nb_lines = dataset.shape[0]
        nb_batch = int(np.ceil(nb_lines / batch_size))
        list_indices = np.arange(nb_lines)
        G_loss1_store = np.zeros(epochs, dtype="float32")  # Cross-entropy loss for G
        G_loss2_store = np.zeros(epochs, dtype="float32")  # RMSE loss for G
        D_loss_store = np.zeros(epochs, dtype="float32")  # Cross-entropy loss for D
        for epoch in range(epochs):
            if (epoch+1) % 1000 == 0:
                print(int((epoch+1)/1000), end=" ", flush=True)
            G_temp1 = []
            G_temp2 = []
            D_temp = []
            np.random.shuffle(list_indices)
            for i in range(nb_batch):
                idx = list_indices[(i * batch_size):((i + 1) * batch_size)]
                cbs = idx.shape[0]  # cbs = current batch size
                planet_batch = dataset[idx]
                G_loss1, G_loss2, D_loss = self.train_step(planet_batch)
                G_temp1.append(G_loss1.numpy() * cbs)  # Standard entropy-loss for G
                G_temp2.append(G_loss2.numpy() * cbs)  # RMSE loss of G (specificity of GAIN)
                D_temp.append(D_loss.numpy() * cbs)  # Standard cross-entropy loss for D
            G_loss1_store[epoch] = np.array(G_temp1).mean() / nb_lines
            G_loss2_store[epoch] = np.array(G_temp2).mean() / nb_lines
            D_loss_store[epoch] = np.array(D_temp).mean() / nb_lines
        print("")
        self.trained = True
        self.nb_epochs = epochs
        self.Gloss1 = G_loss1_store
        self.Gloss2 = G_loss2_store
        self.Dloss = D_loss_store


    def save_weights(self, out_dir):
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        self.G.save(out_dir + "G.h5")
        self.D.save(out_dir + "D.h5")


    def impute(self, nandata):
        noise = tf.random.normal([nandata.shape[0], self.dim])
        M_impute = 1.0 - np.isnan(nandata)
        X_impute = tf.where((M_impute == 0.0), noise, nandata)
        G_input = tf.concat((X_impute, M_impute), axis=1)  # Input of dim 12 (not 6!)
        G_output = self.G(G_input, training=False)
        imputed_data = (X_impute * M_impute + G_output * (1.0 - M_impute)).numpy()
        return imputed_data

