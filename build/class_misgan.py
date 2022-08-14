import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout

from build.utils import maskDistribution, drawMasks
from build.utils import reset_weights


def make_MisGANgen(dim):
    model = Sequential()
    model.add(Dense(128, activation="relu", input_shape=(dim,)))
    model.add(BatchNormalization())
    model.add(Dense(128, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dense(128, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dense(dim, activation="sigmoid"))
    return model


def make_MisGANdisc(dim):
    model = Sequential()
    model.add(Dense(128, activation="relu", input_shape=(dim,)))
    model.add(Dropout(rate=0.3))
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(rate=0.3))
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(rate=0.3))
    model.add(Dense(1, activation="sigmoid"))
    return model


cross_entropy = tf.keras.losses.BinaryCrossentropy()


class MisGAN():

    def __init__(self, dim):
        self.dim = dim
        self.Gx = make_MisGANgen(self.dim)
        self.Dx = make_MisGANdisc(self.dim)
        self.Gi = make_MisGANgen(self.dim)
        self.Di = make_MisGANdisc(self.dim)
        self.Gx_optim = tf.keras.optimizers.Adam(1e-4)
        self.Dx_optim = tf.keras.optimizers.Adam(1e-4)
        self.Gi_optim = tf.keras.optimizers.Adam(1e-4)
        self.Di_optim = tf.keras.optimizers.Adam(1e-4)
        self.unique_masks = []
        self.prob_masks = []
        self.trained = False
        self.nb_epochs = 0
        self.Gx_loss = []
        self.Dx_loss = []
        self.Gi_loss = []
        self.Di_loss = []


    @staticmethod
    def compute_Gx_loss(Dx_fake):
        return cross_entropy(tf.ones_like(Dx_fake), Dx_fake)


    @staticmethod
    def compute_Dx_loss(Dx_real, Dx_fake):
        loss1 = cross_entropy(tf.ones_like(Dx_real), Dx_real)
        loss2 = cross_entropy(tf.zeros_like(Dx_fake), Dx_fake)
        total_loss = loss1 + loss2
        return total_loss


    @staticmethod
    def compute_Gi_loss(Di_fake):
        return cross_entropy(tf.ones_like(Di_fake), Di_fake)


    @staticmethod
    def compute_Di_loss(Di_real, Di_fake):
        loss1 = cross_entropy(tf.ones_like(Di_real), Di_real)
        loss2 = cross_entropy(tf.zeros_like(Di_fake), Di_fake)
        total_loss = loss1 + loss2
        return total_loss


    def reinitialize(self):
        reset_weights(self.Gx)
        reset_weights(self.Dx)
        reset_weights(self.Gi)
        reset_weights(self.Di)
        self.trained = False
        self.nb_epochs = 0
        self.Gx_loss = []
        self.Dx_loss = []
        self.Gi_loss = []
        self.Di_loss = []


    @tf.function  # Causes the function to be "compiled"
    def train_step(self, batch_planets):
        cur_batch_size = batch_planets.shape[0]
        noise1 = tf.random.normal([cur_batch_size, self.dim])
        noise2 = tf.random.normal([cur_batch_size, self.dim])

        with tf.GradientTape() as Gx_tape, tf.GradientTape() as Dx_tape, \
        tf.GradientTape() as Gi_tape, tf.GradientTape() as Di_tape:
            generated_planets = self.Gx(noise1, training=True)
            masks = drawMasks(self.unique_masks, self.prob_masks, cur_batch_size)
            batch_fake = generated_planets * masks
            batch_true = tf.where(tf.math.is_nan(batch_planets), tf.zeros_like(batch_planets), batch_planets)

            Gi_input = tf.where(tf.math.is_nan(batch_planets), noise2, batch_planets)
            Gi_output = self.Gi(Gi_input, training=True)
            imputed_planets = tf.where(tf.math.is_nan(batch_planets), Gi_output, batch_planets)

            Dx_real = self.Dx(batch_true, training=True)
            Dx_fake = self.Dx(batch_fake, training=True)
            Di_real = self.Di(generated_planets, training=True)  # The generated planets are the reference now
            Di_fake = self.Di(imputed_planets, training=True)

            Gx_loss = self.compute_Gx_loss(Dx_fake)
            Dx_loss = self.compute_Dx_loss(Dx_real, Dx_fake)
            Gi_loss = self.compute_Gi_loss(Di_fake)
            Di_loss = self.compute_Di_loss(Di_real, Di_fake)

            Gx_gradients = Gx_tape.gradient(Gx_loss, self.Gx.trainable_variables)
            Dx_gradients = Dx_tape.gradient(Dx_loss, self.Dx.trainable_variables)
            Gi_gradients = Gi_tape.gradient(Gi_loss, self.Gi.trainable_variables)
            Di_gradients = Di_tape.gradient(Di_loss, self.Di.trainable_variables)

            self.Gx_optim.apply_gradients(zip(Gx_gradients, self.Gx.trainable_variables))
            self.Dx_optim.apply_gradients(zip(Dx_gradients, self.Dx.trainable_variables))
            self.Gi_optim.apply_gradients(zip(Gi_gradients, self.Gi.trainable_variables))
            self.Di_optim.apply_gradients(zip(Di_gradients, self.Di.trainable_variables))

            return Gx_loss, Dx_loss, Gi_loss, Di_loss


    def train(self, dataset, batch_size, epochs):
        self.unique_masks, count_masks = maskDistribution(dataset)
        self.prob_masks = count_masks / count_masks.sum()
        nb_lines = dataset.shape[0]
        nb_batch = int(np.ceil(nb_lines / batch_size))
        list_indices = np.arange(nb_lines)
        Gx_loss_store = np.zeros(epochs, dtype="float32")
        Dx_loss_store = np.zeros(epochs, dtype="float32")
        Gi_loss_store = np.zeros(epochs, dtype="float32")
        Di_loss_store = np.zeros(epochs, dtype="float32")
        for epoch in range(epochs):
            if (epoch+1) % 1000 == 0:
                print(int((epoch+1)/1000), end=" ", flush=True)
            Gx_temp = []
            Dx_temp = []
            Gi_temp = []
            Di_temp = []
            np.random.shuffle(list_indices)
            for i in range(nb_batch):
                idx = list_indices[(i * batch_size):((i + 1) * batch_size)]
                cbs = idx.shape[0]  # cbs = current batch size
                planet_batch = dataset[idx]
                Gx_loss, Dx_loss, Gi_loss, Di_loss = self.train_step(planet_batch)
                Gx_temp.append(Gx_loss.numpy() * cbs)
                Dx_temp.append(Dx_loss.numpy() * cbs)
                Gi_temp.append(Gi_loss.numpy() * cbs)
                Di_temp.append(Di_loss.numpy() * cbs)
            Gx_loss_store[epoch] = np.array(Gx_temp).mean() / nb_lines
            Dx_loss_store[epoch] = np.array(Dx_temp).mean() / nb_lines
            Gi_loss_store[epoch] = np.array(Gi_temp).mean() / nb_lines
            Di_loss_store[epoch] = np.array(Di_temp).mean() / nb_lines
        print("")
        self.trained = True
        self.nb_epochs = epochs
        self.Gx_loss = Gx_loss_store
        self.Dx_loss = Dx_loss_store
        self.Gi_loss = Gi_loss_store
        self.Di_loss = Di_loss_store


    def save_weights(self, out_dir):
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        self.Gx.save(out_dir + "Gx.h5")
        self.Dx.save(out_dir + "Dx.h5")
        self.Gi.save(out_dir + "Gi.h5")
        self.Di.save(out_dir + "Di.h5")


    def impute(self, nandata):
        noise = tf.random.normal([nandata.shape[0], self.dim])
        nanmask = tf.math.is_nan(nandata)
        Gi_input = tf.where(nanmask, noise, nandata)
        Gi_output = self.Gi(Gi_input, training=False)
        imputed_data = tf.where(nanmask, Gi_output, nandata).numpy()
        return imputed_data

