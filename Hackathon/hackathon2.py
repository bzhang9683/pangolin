'''
Homework
Re-write your code from hackathon 2 to use convolutional layers and add code to plot a confusion matrix on the validation data.

Specifically, write code to calculate a confusion matrix of the model output on the validation data, and compare to the true labels to calculate a confusion matrix with tf.math.confusion_matrix. (For the inexperienced, what is a confusion matrix?) Use the code example from scikit-learn to help visualise the confusion matrix if you'd like as well.
'''
#%%
import numpy as np                 # to use numpy arrays
import tensorflow as tf            # to specify and run computation graphs
import tensorflow_datasets as tfds # to load training data
import matplotlib.pyplot as plt    # to visualize data and draw plots
from tqdm import tqdm              # to track progress of loops
import time
#%%
# This one iterates through the train data, shuffling and minibatching by 32
train_ds = tfds.load('mnist', split='train[:90%]').shuffle(1024).batch(32)
validation = tfds.load('mnist', split='train[-10%:]',batch_size=-1)
val_ds = tf.cast(validation['image'], tf.float32)
val_labels = validation['label']

#%%
#Hackathon 4
# Create the model
hidden_1 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation=tf.nn.relu, name='hidden_1')
hidden_2 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation=tf.nn.relu, name='hidden_2')
flatten = tf.keras.layers.Flatten()
output = tf.keras.layers.Dense(10)
conv_classifier = tf.keras.Sequential([hidden_1, hidden_2, flatten, output])
optimizer = tf.keras.optimizers.Adam()

train_accu_list = []
# Run some data through the network to initialize it
for batch in train_ds:
    # data is uint8 by default, so we have to cast it
    conv_classifier(tf.cast(batch['image'], tf.float32))
    break
conv_classifier.summary()
for epoch in range(1):
    print('Epoch: ', epoch)
    loss_list = []
    accuracy_train_list = []
    for batch in tqdm(train_ds):
         with tf.GradientTape() as tape:
            # run network
            x = tf.cast(batch['image'], tf.float32)
            labels = batch['label']
            logits = conv_classifier(x)
        
            # calculate loss
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
            loss_list.append(loss)
            # gradient update
            grads = tape.gradient(loss, conv_classifier.trainable_variables)
            optimizer.apply_gradients(zip(grads, conv_classifier.trainable_variables))
        
            # calculate accuracy on the training data set
            predictions = tf.argmax(logits, axis=1)
            accuracy_train = tf.reduce_mean(tf.cast(tf.equal(predictions, labels), tf.float32))
            accuracy_train_list.append(accuracy_train)

    # accuracy and loss on trainning data set
    accu = np.mean(accuracy_train_list)
    #print("Accuracy on Training Dataset:", accu)
    train_accu_list.append(accu)
    #early stop
    if len(train_accu_list) > 1:
        if accu - train_accu_list[epoch-1] < 0.01:
            break
#%%
val_logits = conv_classifier(val_ds)
val_pred = tf.argmax(val_logits, axis=1)
accuracy_val = tf.reduce_mean(tf.cast(tf.equal(val_pred, val_labels), tf.float32))
#%%
tf.math.confusion_matrix(
    val_labels, val_pred, num_classes=10, weights=None, dtype=tf.dtypes.int32,
    name=None
)
# %%
def CalcConfusionMatrix(labels, pred, classes):
    lens = len(classes)
    matrix = np.zeros((lens,lens))
    for i in range(lens):
        c = classes[i]
        for j in range(len(labels)):
            if labels[j] == c:
                if labels[j] == pred[j]:
                    matrix[i,i] +=1
                else:
                    matrix[labels[j],pred[j]] +=1

    return matrix
print(CalcConfusionMatrix(val_labels,val_pred,np.arange(0,10,1)))
    
#%%