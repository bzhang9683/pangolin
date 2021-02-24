import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
def loading_cifar100():
    '''
    Required package: TensorFlow
    '''
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar100.load_data('fine')
    train_ds = tfds.load('cifar100',split='train[:-40%]', batch_size=-1) # If batch_size=-1, will return the full dataset as tf.Tensors.
    X_train = tf.cast(train_ds['image'], tf.float32)/255
    y_train = train_ds['label']
    C = tf.constant(100, name='labels') #the number of labels
    one_hot_train = tf.one_hot(y_train, C)
    y_train_encoded = one_hot_train
    val_ds = tfds.load('cifar100',split='train[-40%:-30%]', batch_size=-1)
    X_val = tf.cast(val_ds['image'], tf.float32)/255
    y_val = val_ds['label']
    one_hot_val = tf.one_hot(y_val, C)
    y_val_encoded = one_hot_val
    test_ds = tfds.load('cifar100',split='train[-30%:]', batch_size=-1)
    X_test = tf.cast(test_ds['image'], tf.float32)/255
    y_test = test_ds['label']
    #one_hot_test = tf.one_hot(y_test, C)
    #y_test_encoded = one_hot_test
    return X_train, y_train_encoded, X_val, y_val_encoded, X_test, y_test

def plot_diagnostics(history):
    #plot loss curve
    plt.subplot(211)
    plt.plot(history.history['loss'], color='blue', label='train')
    plt.plot(history.history['val_loss'], color='red', label='validation')
    plt.title('Cross Entropy Loss')
    #plot accuracy curve
    plt.subplot(212)
    plt.plot(history.history['accuracy'], color='blue', label='train')
    plt.plot(history.history['val_accuracy'], color='red', label='validation')
    plt.title('Classification Accuracy')
