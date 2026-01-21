import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import os 

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

latent_dim = 8

input_size = int(np.load(f'../Simulations/data/input_size.npy'))

tf.random.set_seed(
    1337
)

dummy_input = tf.random.normal((1, input_size))

class Autoencoder(Model):
    """
    Simple feed-forward autoencoder with:
    - Encoder: Flatten -> Dense(16) -> Dense(16) -> Dense(latent_dim)
    - Decoder: Dense(16) -> Dense(16) -> Dense(input_size, sigmoid)
    """
  def __init__(self, latent_dim):
    super(Autoencoder, self).__init__()
    self.latent_dim = latent_dim   
    self.encoder = tf.keras.Sequential([
        layers.Flatten(),
        layers.Dense(16, activation='relu'),
        layers.Dense(16, activation='relu'),
        layers.Dense(latent_dim),
    ], name="encoder"
  )
    self.decoder = tf.keras.Sequential([
        layers.Dense(16, activation='relu'),
        layers.Dense(16, activation='relu'),
        layers.Dense(input_size, activation='sigmoid'),
    ], name="decoder"
  )

  def call(self, x):
    z = self.encoder(x)
    x_prime = self.decoder(z)
    return x_prime
  
  def encode(self, x):
    z = self.encoder(x)
    return z

model = Autoencoder(latent_dim)

model.compile(optimizer='adam', loss=losses.MeanSquaredError(), metrics=['mae'])

model(dummy_input)

