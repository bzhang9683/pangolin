#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf
import numpy as np
import util_FMNIST as util
import model_FMNIST as mf
#from sklearn.model_selection import KFold
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt    # to visualize data and draw plots
from tqdm import tqdm              # to track progress of loops


# ## load the data and run the model 1
# 
# #### The model 1 have one hidden layers. The first layer have 200 neurons. The number of epoch=100, leaning rate = 0.001

# In[2]:


# in order to compare the differnt models and different hyperparameters,
# load the data first and make sure they are using the same split
#load the data from the library
train, val, train_x, train_y, val_x, val_y = util.load_data()

# visualize some of traing data
idx = np.random.randint(train['image'].shape[0])
print("An image looks like this:")
imgplot1 = plt.imshow(train['image'][idx])
print("It's colored because matplotlib wants to provide more contrast than just greys")
plt.show()

# visualize some of traing data
print('validation data shape', val['image'].shape)
idx = np.random.randint(val['image'].shape[0])
print("An image looks like this:")
imgplot1 = plt.imshow(val['image'][idx])
print("It's colored because matplotlib wants to provide more contrast than just greys")
plt.show()


# In[4]:


#with early stopping
#define the parameters
num_trial = 1
num_epoch = 500
verbosity = 1
batch_size = 200
validation_split = 0.3
learning_rate = 0.001

#create the model
model = mf.create_model_1()

#compile the model with optimizer, loss function and metrics.
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'],
    )

#apply simple early stopping
es = EarlyStopping(monitor='val_accuracy', min_delta=0.001, patience = 10)

#check point
#mc = ModelCheckpoint('model', monitor='val_loss', mode='min', verbose=1, save_best_only=True)

#train the data
history = model.fit(
        train_x,train_y,
        batch_size=batch_size,
        epochs = num_epoch,
        verbose=verbosity,
        callbacks = [es],
        validation_split = validation_split)
    
print(model.summary())

util.show_history(history, 'loss', num=num_trial)
util.show_history(history, 'accuracy', num=num_trial)
#test the model
score = model.evaluate(val_x, val_y, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])
#save the score in a text file
np.savetxt(str(num_trial)+ "_score" +".txt", score)

#get the confusion maxtrix
pred_y = model.predict(val_x)
pred_labels = tf.argmax(pred_y, axis = 1)
#pred_labels = to_categorical(pred_y)
con_mat = tf.math.confusion_matrix(val_y, pred_labels, dtype=tf.dtypes.int64, name=None)

#save the confusion maxtrix
con_mat = con_mat.numpy()
fig, ax = plt.subplots(figsize=(10,10))
ax.matshow(con_mat, cmap=plt.cm.Blues)
#show the numbers in the confusion matrix
for i in range(10):
    for j in range(10):
        c = con_mat[j,i]
        ax.text(i, j, str(c), va='center', ha='center')
plt.show()

fig.savefig("./"+ str(num_trial) + '_' + 'model' + ".png")


# In[5]:


#without early stopping
#define the parameters
num_trial = 2
num_epoch = 500
verbosity = 1
validation_split = 0.3
learning_rate = 0.001

#create the model
model = mf.create_model_1()

#compile the model with optimizer, loss function and metrics.
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'],
    )

#apply simple early stopping
#es = EarlyStopping(monitor='val_accuracy', min_delta=0.0001, patience = 25)

#check point
#mc = ModelCheckpoint('model', monitor='val_loss', mode='min', verbose=1, save_best_only=True)

#train the data
history = model.fit(
        train_x,train_y,
        epochs = num_epoch,
        verbose=verbosity,
        validation_split = validation_split)
    
print(model.summary())

util.show_history(history, 'loss', num=num_trial)
util.show_history(history, 'accuracy', num=num_trial)

score = model.evaluate(val_x, val_y, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

#save the score in a text file
np.savetxt(str(num_trial)+ "_score" +".txt", score)

#get the confusion maxtrix
pred_y = model.predict(val_x)
pred_labels = tf.argmax(pred_y, axis = 1)
con_mat = tf.math.confusion_matrix(val_y, pred_labels, dtype=tf.dtypes.int64, name=None)

#save the confusion maxtrix
con_mat = con_mat.numpy()
fig, ax = plt.subplots(figsize=(10,10))
ax.matshow(con_mat, cmap=plt.cm.Blues)
#show the numbers in the confusion matrix
for i in range(10):
    for j in range(10):
        c = con_mat[j,i]
        ax.text(i, j, str(c), va='center', ha='center')
plt.show()

fig.savefig("./"+ str(num_trial) + '_' + 'model' + ".png")


# ## load the data and run the model 2
# 
# #### The model 2 have two layers. The first layer have 200 neurons and the second layer have 20 neurons. The number of epoch=100, leaning rate = 0.001

# In[7]:


#with early stopping
#define the parameters
num_trial = 3
num_epoch = 500
verbosity = 1
validation_split = 0.3
learning_rate = 0.001

#create the model
model = mf.create_model_2()

#compile the model with optimizer, loss function and metrics.
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'],
    )

#apply simple early stopping
es = EarlyStopping(monitor='val_accuracy', min_delta=0.001, patience = 10)

#train the data
history = model.fit(
        train_x,train_y,
        epochs = num_epoch,
        verbose=verbosity,
        callbacks = [es],
        validation_split = validation_split)
    
print(model.summary())

util.show_history(history, 'loss', num=num_trial)
util.show_history(history, 'accuracy', num=num_trial)

score = model.evaluate(val_x, val_y, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

#save the score in a text file
np.savetxt(str(num_trial)+ "_score" +".txt", score)

#get the confusion maxtrix
pred_y = model.predict(val_x)
pred_labels = tf.argmax(pred_y, axis = 1)
con_mat = tf.math.confusion_matrix(val_y, pred_labels, dtype=tf.dtypes.int64, name=None)

#save the confusion maxtrix
con_mat = con_mat.numpy()
fig, ax = plt.subplots(figsize=(10,10))
ax.matshow(con_mat, cmap=plt.cm.Blues)
#show the numbers in the confusion matrix
for i in range(10):
    for j in range(10):
        c = con_mat[j,i]
        ax.text(i, j, str(c), va='center', ha='center')
plt.show()

fig.savefig("./"+ str(num_trial) + '_' + 'model' + ".png")


# In[8]:


#without early stopping
#define the parameters
num_trial = 4
num_epoch = 500
verbosity = 1
validation_split = 0.3
learning_rate = 0.001

#create the model
model = mf.create_model_2()

#compile the model with optimizer, loss function and metrics.
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'],
    )

#apply simple early stopping
#es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15)

#check point
#mc = ModelCheckpoint('model', monitor='val_loss', mode='min', verbose=1, save_best_only=True)

#train the data
history = model.fit(
        train_x,train_y,
        epochs = num_epoch,
        verbose=verbosity,
        validation_split = validation_split)
    
print(model.summary())

util.show_history(history, 'loss', num=num_trial)
util.show_history(history, 'accuracy', num=num_trial)

score = model.evaluate(val_x, val_y, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

#save the score in a text file
np.savetxt(str(num_trial)+ "_score" +".txt", score)

#get the confusion maxtrix
pred_y = model.predict(val_x)
pred_labels = tf.argmax(pred_y, axis = 1)
con_mat = tf.math.confusion_matrix(val_y, pred_labels, dtype=tf.dtypes.int64, name=None)

#save the confusion maxtrix
con_mat = con_mat.numpy()
fig, ax = plt.subplots(figsize=(10,10))
ax.matshow(con_mat, cmap=plt.cm.Blues)
#show the numbers in the confusion matrix
for i in range(10):
    for j in range(10):
        c = con_mat[j,i]
        ax.text(i, j, str(c), va='center', ha='center')
plt.show()

fig.savefig("./"+ str(num_trial) + '_' + 'model' + ".png")


# ## load the data and run the model 3
# 
# #### The model 3 have three layers. The first layer have 200 neurons, the second layer has 20 neurons and the third has 20 neurons. The number of epoch=100, leaning rate = 0.001

# In[ ]:


#with early stopping
#define the parameters
num_trial = 5
num_epoch = 500
verbosity = 1
validation_split = 0.3
learning_rate = 0.001

#create the model
model = mf.create_model_3()

#compile the model with optimizer, loss function and metrics.
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'],
    )

#apply simple early stopping
es = EarlyStopping(monitor='val_accuracy', min_delta=0.001, patience = 10)

#train the data
history = model.fit(
        train_x,train_y,
        epochs = num_epoch,
        verbose=verbosity,
        callbacks = [es],
        validation_split = validation_split)
    
print(model.summary())

util.show_history(history, 'loss', num=num_trial)
util.show_history(history, 'accuracy', num=num_trial)

score = model.evaluate(val_x, val_y, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

#save the score in a text file
np.savetxt(str(num_trial)+ "_score" +".txt", score)

#get the confusion maxtrix
pred_y = model.predict(val_x)
pred_labels = tf.argmax(pred_y, axis = 1)
con_mat = tf.math.confusion_matrix(val_y, pred_labels, dtype=tf.dtypes.int64, name=None)

#save the confusion maxtrix
con_mat = con_mat.numpy()
fig, ax = plt.subplots(figsize=(10,10))
ax.matshow(con_mat, cmap=plt.cm.Blues)
#show the numbers in the confusion matrix
for i in range(10):
    for j in range(10):
        c = con_mat[j,i]
        ax.text(i, j, str(c), va='center', ha='center')
plt.show()

fig.savefig("./"+ str(num_trial) + '_' + 'model' + ".png")


# In[ ]:


#without early stopping
#define the parameters
num_trial = 6
num_epoch = 500
verbosity = 1
validation_split = 0.3
learning_rate = 0.001

#create the model
model = mf.create_model_3()

#compile the model with optimizer, loss function and metrics.
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'],
    )

#apply simple early stopping
#es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15)

#check point
#mc = ModelCheckpoint('model', monitor='val_loss', mode='min', verbose=1, save_best_only=True)

#train the data
history = model.fit(
        train_x,train_y,
        epochs = num_epoch,
        verbose=verbosity,
        validation_split = validation_split)
    
print(model.summary())

util.show_history(history, 'loss', num=num_trial)
util.show_history(history, 'accuracy', num=num_trial)

score = model.evaluate(val_x, val_y, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

#save the score in a text file
np.savetxt(str(num_trial)+ "_score" +".txt", score)

#get the confusion maxtrix
pred_y = model.predict(val_x)
pred_labels = tf.argmax(pred_y, axis = 1)
con_mat = tf.math.confusion_matrix(val_y, pred_labels, dtype=tf.dtypes.int64, name=None)

#save the confusion maxtrix
con_mat = con_mat.numpy()
fig, ax = plt.subplots(figsize=(10,10))
ax.matshow(con_mat, cmap=plt.cm.Blues)
#show the numbers in the confusion matrix
for i in range(10):
    for j in range(10):
        c = con_mat[j,i]
        ax.text(i, j, str(c), va='center', ha='center')
plt.show()

fig.savefig("./"+ str(num_trial) + '_' + 'model' + ".png")


# ## load the data and run the model 4
# 
# #### The model 4 have two layers. The first layer have 400 neurons, the second layer has 20 neurons. The number of epoch=100, leaning rate = 0.001

# In[ ]:


#with early stopping
#define the parameters
num_trial = 7
num_epoch = 500
verbosity = 1
validation_split = 0.3
learning_rate = 0.001

#create the model
model = mf.create_model_4()

#compile the model with optimizer, loss function and metrics.
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'],
    )

#apply simple early stopping
es = EarlyStopping(monitor='val_accuracy', min_delta=0.001, patience = 10)

#train the data
history = model.fit(
        train_x,train_y,
        epochs = num_epoch,
        verbose=verbosity,
        callbacks = [es],
        validation_split = validation_split)
    
print(model.summary())

util.show_history(history, 'loss', num=num_trial)
util.show_history(history, 'accuracy', num=num_trial)

score = model.evaluate(val_x, val_y, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

#save the score in a text file
np.savetxt(str(num_trial)+ "_score" +".txt", score)

#get the confusion maxtrix
pred_y = model.predict(val_x)
pred_labels = tf.argmax(pred_y, axis = 1)
con_mat = tf.math.confusion_matrix(val_y, pred_labels, dtype=tf.dtypes.int64, name=None)

#save the confusion maxtrix
con_mat = con_mat.numpy()
fig, ax = plt.subplots(figsize=(10,10))
ax.matshow(con_mat, cmap=plt.cm.Blues)
#show the numbers in the confusion matrix
for i in range(10):
    for j in range(10):
        c = con_mat[j,i]
        ax.text(i, j, str(c), va='center', ha='center')
plt.show()

fig.savefig("./"+ str(num_trial) + '_' + 'model' + ".png")


# In[ ]:


#without early stopping
#define the parameters
num_trial = 8
num_epoch = 500
verbosity = 1
validation_split = 0.3
learning_rate = 0.001

#create the model
model = mf.create_model_4()

#compile the model with optimizer, loss function and metrics.
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'],
    )

#apply simple early stopping
#es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15)

#check point
#mc = ModelCheckpoint('model', monitor='val_loss', mode='min', verbose=1, save_best_only=True)

#train the data
history = model.fit(
        train_x,train_y,
        epochs = num_epoch,
        verbose=verbosity,
        validation_split = validation_split)
    
print(model.summary())

util.show_history(history, 'loss', num=num_trial)
util.show_history(history, 'accuracy', num=num_trial)

score = model.evaluate(val_x, val_y, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

#save the score in a text file
np.savetxt(str(num_trial)+ "_score" +".txt", score)

#get the confusion maxtrix
pred_y = model.predict(val_x)
pred_labels = tf.argmax(pred_y, axis = 1)
con_mat = tf.math.confusion_matrix(val_y, pred_labels, dtype=tf.dtypes.int64, name=None)

#save the confusion maxtrix
con_mat = con_mat.numpy()
fig, ax = plt.subplots(figsize=(10,10))
ax.matshow(con_mat, cmap=plt.cm.Blues)
#show the numbers in the confusion matrix
for i in range(10):
    for j in range(10):
        c = con_mat[j,i]
        ax.text(i, j, str(c), va='center', ha='center')
plt.show()

fig.savefig("./"+ str(num_trial) + '_' + 'model' + ".png")


# ## load the data and run the model 5
# 
# #### The model 5 have two layers. The first layer have 800 neurons, the second layer has 20 neurons. The number of epoch=100, leaning rate = 0.001

# In[10]:


#with early stopping
#define the parameters
num_trial = 9
num_epoch = 500
verbosity = 1
validation_split = 0.3
learning_rate = 0.001

#create the model
model = mf.create_model_5()

#compile the model with optimizer, loss function and metrics.
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'],
    )

#apply simple early stopping
es = EarlyStopping(monitor='val_accuracy', min_delta=0.001, patience = 10)

#train the data
history = model.fit(
        train_x,train_y,
        epochs = num_epoch,
        verbose=verbosity,
        callbacks = [es],
        validation_split = validation_split)
    
print(model.summary())

util.show_history(history, 'loss', num=num_trial)
util.show_history(history, 'accuracy', num=num_trial)

score = model.evaluate(val_x, val_y, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

#save the score in a text file
np.savetxt(str(num_trial)+ "_score" +".txt", score)

#get the confusion maxtrix
pred_y = model.predict(val_x)
pred_labels = tf.argmax(pred_y, axis = 1)
con_mat = tf.math.confusion_matrix(val_y, pred_labels, dtype=tf.dtypes.int64, name=None)

#save the confusion maxtrix
con_mat = con_mat.numpy()
fig, ax = plt.subplots(figsize=(10,10))
ax.matshow(con_mat, cmap=plt.cm.Blues)
#show the numbers in the confusion matrix
for i in range(10):
    for j in range(10):
        c = con_mat[j,i]
        ax.text(i, j, str(c), va='center', ha='center')
plt.show()

fig.savefig("./"+ str(num_trial) + '_' + 'model' + ".png")


# In[ ]:


#without early stopping
#define the parameters
num_trial = 10
num_epoch = 500
verbosity = 1
validation_split = 0.3
learning_rate = 0.001

#create the model
model = mf.create_model_5()

#compile the model with optimizer, loss function and metrics.
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'],
    )

#apply simple early stopping
#es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15)

#check point
#mc = ModelCheckpoint('model', monitor='val_loss', mode='min', verbose=1, save_best_only=True)

#train the data
history = model.fit(
        train_x,train_y,
        epochs = num_epoch,
        verbose=verbosity,
        validation_split = validation_split)
    
print(model.summary())

util.show_history(history, 'loss', num=num_trial)
util.show_history(history, 'accuracy', num=num_trial)

score = model.evaluate(val_x, val_y, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

#save the score in a text file
np.savetxt(str(num_trial)+ "_score" +".txt", score)

#get the confusion maxtrix
pred_y = model.predict(val_x)
pred_labels = tf.argmax(pred_y, axis = 1)
con_mat = tf.math.confusion_matrix(val_y, pred_labels, dtype=tf.dtypes.int64, name=None)

#save the confusion maxtrix
con_mat = con_mat.numpy()
fig, ax = plt.subplots(figsize=(10,10))
ax.matshow(con_mat, cmap=plt.cm.Blues)
#show the numbers in the confusion matrix
for i in range(10):
    for j in range(10):
        c = con_mat[j,i]
        ax.text(i, j, str(c), va='center', ha='center')
plt.show()

fig.savefig("./"+ str(num_trial) + '_' + 'model' + ".png")


# ## load the data and run the model 6
# 
# #### The model 6 have two layers. The first layer have 1600 neurons, the second layer has 20 neurons. The number of epoch=100, leaning rate = 0.001

# In[ ]:


#with early stopping
#define the parameters
num_trial = 11
num_epoch = 500
verbosity = 1
validation_split = 0.3
learning_rate = 0.001

#create the model
model = mf.create_model_6()

#compile the model with optimizer, loss function and metrics.
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'],
    )

#apply simple early stopping
es = EarlyStopping(monitor='val_accuracy', min_delta=0.001, patience = 10)

#train the data
history = model.fit(
        train_x,train_y,
        epochs = num_epoch,
        verbose=verbosity,
        callbacks = [es],
        validation_split = validation_split)
    
print(model.summary())

util.show_history(history, 'loss', num=num_trial)
util.show_history(history, 'accuracy', num=num_trial)

score = model.evaluate(val_x, val_y, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

#save the score in a text file
np.savetxt(str(num_trial)+ "_score" +".txt", score)

#get the confusion maxtrix
pred_y = model.predict(val_x)
pred_labels = tf.argmax(pred_y, axis = 1)
con_mat = tf.math.confusion_matrix(val_y, pred_labels, dtype=tf.dtypes.int64, name=None)

#save the confusion maxtrix
con_mat = con_mat.numpy()
fig, ax = plt.subplots(figsize=(10,10))
ax.matshow(con_mat, cmap=plt.cm.Blues)
#show the numbers in the confusion matrix
for i in range(10):
    for j in range(10):
        c = con_mat[j,i]
        ax.text(i, j, str(c), va='center', ha='center')
plt.show()

fig.savefig("./"+ str(num_trial) + '_' + 'model' + ".png")


# In[ ]:


#without early stopping
#define the parameters
num_trial = 12
num_epoch = 500
verbosity = 1
validation_split = 0.3
learning_rate = 0.001

#create the model
model = mf.create_model_6()

#compile the model with optimizer, loss function and metrics.
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'],
    )

#apply simple early stopping
#es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15)

#check point
#mc = ModelCheckpoint('model', monitor='val_loss', mode='min', verbose=1, save_best_only=True)

#train the data
history = model.fit(
        train_x,train_y,
        epochs = num_epoch,
        verbose=verbosity,
        validation_split = validation_split)
    
print(model.summary())

util.show_history(history, 'loss', num=num_trial)
util.show_history(history, 'accuracy', num=num_trial)

score = model.evaluate(val_x, val_y, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

#save the score in a text file
np.savetxt(str(num_trial)+ "_score" +".txt", score)

#get the confusion maxtrix
pred_y = model.predict(val_x)
pred_labels = tf.argmax(pred_y, axis = 1)
con_mat = tf.math.confusion_matrix(val_y, pred_labels, dtype=tf.dtypes.int64, name=None)

#save the confusion maxtrix
con_mat = con_mat.numpy()
fig, ax = plt.subplots(figsize=(10,10))
ax.matshow(con_mat, cmap=plt.cm.Blues)
#show the numbers in the confusion matrix
for i in range(10):
    for j in range(10):
        c = con_mat[j,i]
        ax.text(i, j, str(c), va='center', ha='center')
plt.show()

fig.savefig("./"+ str(num_trial) + '_' + 'model' + ".png")


# ## load the data and run the model 2
# 
# #### The model 2 have two layers. The first layer have 200 neurons and the second layer have 20 neurons. The number of epoch=100, leaning rate = 0.01

# In[ ]:


#with early stopping
#define the parameters
num_trial = 13
num_epoch = 500
verbosity = 1
validation_split = 0.3
learning_rate = 0.01

#create the model
model = mf.create_model_2()

#compile the model with optimizer, loss function and metrics.
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'],
    )

#apply simple early stopping
es = EarlyStopping(monitor='val_accuracy', min_delta=0.001, patience = 10)

#train the data
history = model.fit(
        train_x,train_y,
        epochs = num_epoch,
        verbose=verbosity,
        callbacks = [es],
        validation_split = validation_split)
    
print(model.summary())

util.show_history(history, 'loss', num=num_trial)
util.show_history(history, 'accuracy', num=num_trial)

score = model.evaluate(val_x, val_y, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

#save the score in a text file
np.savetxt(str(num_trial)+ "_score" +".txt", score)

#get the confusion maxtrix
pred_y = model.predict(val_x)
pred_labels = tf.argmax(pred_y, axis = 1)
con_mat = tf.math.confusion_matrix(val_y, pred_labels, dtype=tf.dtypes.int64, name=None)

#save the confusion maxtrix
con_mat = con_mat.numpy()
fig, ax = plt.subplots(figsize=(10,10))
ax.matshow(con_mat, cmap=plt.cm.Blues)
#show the numbers in the confusion matrix
for i in range(10):
    for j in range(10):
        c = con_mat[j,i]
        ax.text(i, j, str(c), va='center', ha='center')
plt.show()

fig.savefig("./"+ str(num_trial) + '_' + 'model' + ".png")


# In[ ]:


#without early stopping
#define the parameters
num_trial = 14
num_epoch = 500
verbosity = 1
validation_split = 0.3
learning_rate = 0.01

#create the model
model = mf.create_model_2()

#compile the model with optimizer, loss function and metrics.
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'],
    )

#apply simple early stopping
#es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15)

#check point
#mc = ModelCheckpoint('model', monitor='val_loss', mode='min', verbose=1, save_best_only=True)

#train the data
history = model.fit(
        train_x,train_y,
        epochs = num_epoch,
        verbose=verbosity,
        validation_split = validation_split)
    
print(model.summary())

util.show_history(history, 'loss', num=num_trial)
util.show_history(history, 'accuracy', num=num_trial)

score = model.evaluate(val_x, val_y, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

#save the score in a text file
np.savetxt(str(num_trial)+ "_score" +".txt", score)

#get the confusion maxtrix
pred_y = model.predict(val_x)
pred_labels = tf.argmax(pred_y, axis = 1)
con_mat = tf.math.confusion_matrix(val_y, pred_labels, dtype=tf.dtypes.int64, name=None)

#save the confusion maxtrix
con_mat = con_mat.numpy()
fig, ax = plt.subplots(figsize=(10,10))
ax.matshow(con_mat, cmap=plt.cm.Blues)
#show the numbers in the confusion matrix
for i in range(10):
    for j in range(10):
        c = con_mat[j,i]
        ax.text(i, j, str(c), va='center', ha='center')
plt.show()

fig.savefig("./"+ str(num_trial) + '_' + 'model' + ".png")


# ## load the data and run the model 2
# 
# #### The model 2 have two layers. The first layer have 200 neurons and the second layer have 20 neurons. The number of epoch=100, leaning rate = 0.5

# In[ ]:


#with early stopping
#define the parameters
num_trial = 15
num_epoch = 500
verbosity = 1
validation_split = 0.3
learning_rate = 0.5

#create the model
model = mf.create_model_2()

#compile the model with optimizer, loss function and metrics.
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'],
    )

#apply simple early stopping
es = EarlyStopping(monitor='val_accuracy', min_delta=0.001, patience = 10)

#train the data
history = model.fit(
        train_x,train_y,
        epochs = num_epoch,
        verbose=verbosity,
        callbacks = [es],
        validation_split = validation_split)
    
print(model.summary())

util.show_history(history, 'loss', num=num_trial)
util.show_history(history, 'accuracy', num=num_trial)

score = model.evaluate(val_x, val_y, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

#save the score in a text file
np.savetxt(str(num_trial)+ "_score" +".txt", score)

#get the confusion maxtrix
pred_y = model.predict(val_x)
pred_labels = tf.argmax(pred_y, axis = 1)
con_mat = tf.math.confusion_matrix(val_y, pred_labels, dtype=tf.dtypes.int64, name=None)

#save the confusion maxtrix
con_mat = con_mat.numpy()
fig, ax = plt.subplots(figsize=(10,10))
ax.matshow(con_mat, cmap=plt.cm.Blues)
#show the numbers in the confusion matrix
for i in range(10):
    for j in range(10):
        c = con_mat[j,i]
        ax.text(i, j, str(c), va='center', ha='center')
plt.show()

fig.savefig("./"+ str(num_trial) + '_' + 'model' + ".png")


# In[ ]:


#without early stopping
#define the parameters
num_trial = 16
num_epoch = 500
verbosity = 1
validation_split = 0.3
learning_rate = 0.5

#create the model
model = mf.create_model_2()

#compile the model with optimizer, loss function and metrics.
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'],
    )

#apply simple early stopping
#es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15)

#check point
#mc = ModelCheckpoint('model', monitor='val_loss', mode='min', verbose=1, save_best_only=True)

#train the data
history = model.fit(
        train_x,train_y,
        epochs = num_epoch,
        verbose=verbosity,
        validation_split = validation_split)
    
print(model.summary())

util.show_history(history, 'loss', num=num_trial)
util.show_history(history, 'accuracy', num=num_trial)

score = model.evaluate(val_x, val_y, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

#save the score in a text file
np.savetxt(str(num_trial)+ "_score" +".txt", score)

#get the confusion maxtrix
pred_y = model.predict(val_x)
pred_labels = tf.argmax(pred_y, axis = 1)
con_mat = tf.math.confusion_matrix(val_y, pred_labels, dtype=tf.dtypes.int64, name=None)

#save the confusion maxtrix
con_mat = con_mat.numpy()
fig, ax = plt.subplots(figsize=(10,10))
ax.matshow(con_mat, cmap=plt.cm.Blues)
#show the numbers in the confusion matrix
for i in range(10):
    for j in range(10):
        c = con_mat[j,i]
        ax.text(i, j, str(c), va='center', ha='center')
plt.show()

fig.savefig("./"+ str(num_trial) + '_' + 'model' + ".png")


# ## load the data and run the model 3
# 
# #### The model 3 have three layers. The first layer have 200 neurons, the second layer has 20 neurons and the third has 20 neurons. The number of epoch=100, leaning rate = 0.01

# In[ ]:


#with early stopping
#define the parameters
num_trial = 17
num_epoch = 500
verbosity = 1
validation_split = 0.3
learning_rate = 0.01

#create the model
model = mf.create_model_3()

#compile the model with optimizer, loss function and metrics.
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'],
    )

#apply simple early stopping
es = EarlyStopping(monitor='val_accuracy', min_delta=0.001, patience = 10)

#train the data
history = model.fit(
        train_x,train_y,
        epochs = num_epoch,
        verbose=verbosity,
        callbacks = [es],
        validation_split = validation_split)
    
print(model.summary())

util.show_history(history, 'loss', num=num_trial)
util.show_history(history, 'accuracy', num=num_trial)

score = model.evaluate(val_x, val_y, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

#save the score in a text file
np.savetxt(str(num_trial)+ "_score" +".txt", score)

#get the confusion maxtrix
pred_y = model.predict(val_x)
pred_labels = tf.argmax(pred_y, axis = 1)
con_mat = tf.math.confusion_matrix(val_y, pred_labels, dtype=tf.dtypes.int64, name=None)

#save the confusion maxtrix
con_mat = con_mat.numpy()
fig, ax = plt.subplots(figsize=(10,10))
ax.matshow(con_mat, cmap=plt.cm.Blues)
#show the numbers in the confusion matrix
for i in range(10):
    for j in range(10):
        c = con_mat[j,i]
        ax.text(i, j, str(c), va='center', ha='center')
plt.show()

fig.savefig("./"+ str(num_trial) + '_' + 'model' + ".png")


# In[ ]:


#without early stopping
#define the parameters
num_trial = 18
num_epoch = 500
verbosity = 1
validation_split = 0.3
learning_rate = 0.01

#create the model
model = mf.create_model_3()

#compile the model with optimizer, loss function and metrics.
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'],
    )

#apply simple early stopping
#es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15)

#check point
#mc = ModelCheckpoint('model', monitor='val_loss', mode='min', verbose=1, save_best_only=True)

#train the data
history = model.fit(
        train_x,train_y,
        epochs = num_epoch,
        verbose=verbosity,
        validation_split = validation_split)
    
print(model.summary())

util.show_history(history, 'loss', num=num_trial)
util.show_history(history, 'accuracy', num=num_trial)

score = model.evaluate(val_x, val_y, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

#save the score in a text file
np.savetxt(str(num_trial)+ "_score" +".txt", score)

#get the confusion maxtrix
pred_y = model.predict(val_x)
pred_labels = tf.argmax(pred_y, axis = 1)
con_mat = tf.math.confusion_matrix(val_y, pred_labels, dtype=tf.dtypes.int64, name=None)

#save the confusion maxtrix
con_mat = con_mat.numpy()
fig, ax = plt.subplots(figsize=(10,10))
ax.matshow(con_mat, cmap=plt.cm.Blues)
#show the numbers in the confusion matrix
for i in range(10):
    for j in range(10):
        c = con_mat[j,i]
        ax.text(i, j, str(c), va='center', ha='center')
plt.show()

fig.savefig("./"+ str(num_trial) + '_' + 'model' + ".png")


# ## load the data and run the model 3
# 
# #### The model 3 have three layers. The first layer have 200 neurons, the second layer has 20 neurons and the third has 20 neurons. The number of epoch=100, leaning rate = 0.5

# In[ ]:


#with early stopping
#define the parameters
num_trial = 19
num_epoch = 500
verbosity = 1
validation_split = 0.3
learning_rate = 0.5

#create the model
model = mf.create_model_3()

#compile the model with optimizer, loss function and metrics.
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'],
    )

#apply simple early stopping
es = EarlyStopping(monitor='val_accuracy', min_delta=0.001, patience = 10)

#train the data
history = model.fit(
        train_x,train_y,
        epochs = num_epoch,
        verbose=verbosity,
        callbacks = [es],
        validation_split = validation_split)
    
print(model.summary())

util.show_history(history, 'loss', num=num_trial)
util.show_history(history, 'accuracy', num=num_trial)

score = model.evaluate(val_x, val_y, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

#save the score in a text file
np.savetxt(str(num_trial)+ "_score" +".txt", score)

#get the confusion maxtrix
pred_y = model.predict(val_x)
pred_labels = tf.argmax(pred_y, axis = 1)
con_mat = tf.math.confusion_matrix(val_y, pred_labels, dtype=tf.dtypes.int64, name=None)

#save the confusion maxtrix
con_mat = con_mat.numpy()
fig, ax = plt.subplots(figsize=(10,10))
ax.matshow(con_mat, cmap=plt.cm.Blues)
#show the numbers in the confusion matrix
for i in range(10):
    for j in range(10):
        c = con_mat[j,i]
        ax.text(i, j, str(c), va='center', ha='center')
plt.show()

fig.savefig("./"+ str(num_trial) + '_' + 'model' + ".png")


# In[ ]:


#without early stopping
#define the parameters
num_trial = 20
num_epoch = 500
verbosity = 1
validation_split = 0.3
learning_rate = 0.5

#create the model
model = mf.create_model_3()

#compile the model with optimizer, loss function and metrics.
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'],
    )

#apply simple early stopping
#es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15)

#check point
#mc = ModelCheckpoint('model', monitor='val_loss', mode='min', verbose=1, save_best_only=True)

#train the data
history = model.fit(
        train_x,train_y,
        epochs = num_epoch,
        verbose=verbosity,
        validation_split = validation_split)
    
print(model.summary())

util.show_history(history, 'loss', num=num_trial)
util.show_history(history, 'accuracy', num=num_trial)

score = model.evaluate(val_x, val_y, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

#save the score in a text file
np.savetxt(str(num_trial)+ "_score" +".txt", score)

#get the confusion maxtrix
pred_y = model.predict(val_x)
pred_labels = tf.argmax(pred_y, axis = 1)
con_mat = tf.math.confusion_matrix(val_y, pred_labels, dtype=tf.dtypes.int64, name=None)

#save the confusion maxtrix
con_mat = con_mat.numpy()
fig, ax = plt.subplots(figsize=(10,10))
ax.matshow(con_mat, cmap=plt.cm.Blues)
#show the numbers in the confusion matrix
for i in range(10):
    for j in range(10):
        c = con_mat[j,i]
        ax.text(i, j, str(c), va='center', ha='center')
plt.show()

fig.savefig("./"+ str(num_trial) + '_' + 'model' + ".png")

