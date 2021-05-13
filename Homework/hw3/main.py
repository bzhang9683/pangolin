#!/usr/bin/env python
# coding: utf-8

# In[1]:


import util
import model


# In[2]:


#load the dataset
train_dataset, valid_dataset = util.load_data()


# In[3]:


#show the image in the dataset
util.show_image(train_dataset, 1)


# In[4]:


latent_dim = 200
n_epochs = 10
batch_size = 32


# In[5]:


#define the models
#create the discriminator
discriminator = model.define_discriminator()
# create the generator
generator = model.define_generator(latent_dim)
# create the gan
gan_model = model.define_gan(generator, discriminator)

discriminator.summary()
generator.summary()

gan_model.summary()


# In[6]:


# train model
util.train(generator, discriminator, gan_model, train_dataset, latent_dim, n_epochs=n_epochs, n_batch=batch_size)


# In[6]:


# load model
model = tf.keras.models.load_model('/work/cse479/binma/Hw3/1/saved_model')
# generate images
latent_points = generate_latent_points(200, 64)
# generate images
X = model.predict(latent_points)
# plot the result
show_plot(X, 1)

