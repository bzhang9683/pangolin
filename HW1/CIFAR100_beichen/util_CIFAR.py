import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np

def loading_cifar100():
    '''
    Required package: TensorFlow
    '''
    (train_ds,val_ds,test_ds) = tfds.load(name='cifar100',split=['train[:-20%]','train[-20%:-10%]','train[-10%:]'], batch_size=-1)
    # If batch_size=-1, will return the full dataset as tf.Tensors.
    X_train = tf.cast(train_ds['image'], tf.float32)/255
    y_train = train_ds['label']
    C = tf.constant(100, name='labels') #the number of labels
    one_hot_train = tf.one_hot(y_train, C)
    y_train_encoded = one_hot_train
    #val_ds = tfds.load(name = 'cifar100',split='train[-30%:-20%]', batch_size=-1, as_supervised=True)
    X_val = tf.cast(val_ds['image'], tf.float32)/255
    y_val = val_ds['label']
    one_hot_val = tf.one_hot(y_val, C)
    y_val_encoded = one_hot_val
    #test_ds = tfds.load(name = 'cifar100',split='train[-20%:]', batch_size=-1, as_supervised=True)
    X_test = tf.cast(test_ds['image'], tf.float32)/255
    y_test = test_ds['label']
    #one_hot_test = tf.one_hot(y_test, C)
    #y_test_encoded = one_hot_test
    return X_train, y_train_encoded, X_val, y_val_encoded, X_test, y_test

def plot_diagnostics(history):
    #plot loss curve
    plt.figure()
    plt.plot(history.history['loss'], color='blue', label='train')
    plt.plot(history.history['val_loss'], color='red', label='validation')
    plt.legend()
    plt.title('Cross Entropy Loss')
    plt.savefig('loss_curve.png')
    #plot accuracy curve
    plt.figure()
    plt.plot(history.history['accuracy'], color='blue', label='train')
    plt.plot(history.history['val_accuracy'], color='red', label='validation')
    plt.legend()
    plt.title('Classification Accuracy')
    plt.savefig('accu_curve.png')

def confidence_interval(error,n):
    low = error -1.96 *np.sqrt((error * (1-error)) / n)
    high = error + 1.96 *np.sqrt((error * (1-error)) / n)
    return low, high