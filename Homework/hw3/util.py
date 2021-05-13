#!/usr/bin/env python
# coding: utf-8

# In[13]:


import os
import glob
import pathlib
from numpy.random import randn
from numpy.random import randint
from numpy import expand_dims
from numpy import zeros
from numpy import ones

import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tqdm import tqdm


# In[7]:
n_epochs = 10
batch_size = 32
dropout_rate = 0.3
        
validation_rate = 0.2
    
image_size_1 = 218
image_size_2 = 178
channels = 3

data_root_orig = "/Users/mabin/Documents/phD_life/3-PhD/Classes/CSCE/879/project/homework03/archive/img_align_celeba/img_align_celeba/"


# In[5]:


def build_decoder(image_size_1, image_size_2, channels,test=False, ):
    def decoder(path):
        img = file_bytes = tf.io.read_file(path)
        img = tf.image.decode_jpeg(file_bytes, channels=channels)  
        img = tf.image.resize(img, (image_size_1, image_size_2))
        img = tf.cast(img, tf.float32) / 255.0
        return img
    def decoder_train(path):
        return decoder(path), 1

    return decoder if test else decoder_train


# In[6]:


def build_dataset(paths, test=False, shuffle=1, batch_size=10):
    AUTO = tf.data.experimental.AUTOTUNE
    decoder = build_decoder(image_size_1, image_size_2, channels,test)

    dset = tf.data.Dataset.from_tensor_slices(paths)
    dset = dset.map(decoder, num_parallel_calls=AUTO)
    
    dset = dset.shuffle(shuffle)
    dset = dset.batch(batch_size)
    return dset


# In[8]:


def load_data():
    data_root = pathlib.Path(data_root_orig)
    all_image_paths = list(data_root.glob('*'))
    all_image_paths = [str(path) for path in all_image_paths]

    train_paths, valid_paths, _, _ = train_test_split(all_image_paths, all_image_paths, test_size=validation_rate, shuffle=True)

    train_dataset = build_dataset(train_paths, batch_size=batch_size)
    valid_dataset = build_dataset(valid_paths, batch_size=batch_size)
    
    return train_dataset, valid_dataset


# In[9]:


def show_image(dataset, num_images):
    plt.figure(figsize=(4,4))
    for n, (image, label) in enumerate(dataset.unbatch().take(num_images)):       
        f, ax1 = plt.subplots(1) 
        ax1.imshow(image)
        print(label)
        plt.show()


# In[11]:


def generate_latent_points(latent_dim, n_samples):
    # generate points in the latent space
    x_input = randn(latent_dim * n_samples)
    # reshape into a batch of inputs for the network
    x_input = x_input.reshape(n_samples, latent_dim)
    return x_input


# In[ ]:


def generate_fake_samples(generator, latent_dim, n_samples):
    # generate points in latent space
    x_input = generate_latent_points(latent_dim, n_samples)
    # predict outputs
    X = generator.predict(x_input)
    # create class labels
    y = zeros((n_samples, 1))
    return X, y


# In[ ]:


def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=n_epochs, n_batch=batch_size):
    half_batch = int(n_batch / 2)
    # manually enumerate epochs
    for i in range(n_epochs):
            # enumerate batches over the training set
        for batch in tqdm(dataset):
            # get randomly selected 'real' samples
            X_real = batch[0]
            y_real = batch[1]
            # update discriminator model weights
            d_loss1, _ = d_model.train_on_batch(X_real, y_real)
            # generate 'fake' examples
            X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
            # update discriminator model weights
            d_loss2, _ = d_model.train_on_batch(X_fake, y_fake)
            # prepare points in latent space as input for the generator
            X_gan = generate_latent_points(latent_dim, n_batch)
            # create inverted labels for the fake samples
            y_gan = ones((n_batch, 1))
            # update the generator via the discriminator's error
            g_loss = gan_model.train_on_batch(X_gan, y_gan)
    
    # save the generator model
    g_model.complie()
    tf.saved_model.save(g_model, '/work/cse479/binma/Hw3/1/saved_model')

