from __future__ import print_function
import tensorflow as tf            # to specify and run computation graphs
import tensorflow_datasets as tfds # to load training data
import matplotlib.pyplot as plt    # to visualize data and draw plots
from tensorflow.keras.layers import Dense, Flatten

def create_model_1():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(10, tf.nn.relu))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))
    
    return model

def create_model_2():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(20, tf.nn.relu))
    model.add(tf.keras.layers.Dense(10, tf.nn.relu))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))
    
    return model

def create_model_3():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(200, tf.nn.relu))
    model.add(tf.keras.layers.Dense(20, tf.nn.relu))
    model.add(tf.keras.layers.Dense(10, tf.nn.relu))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))
    
    return model

def create_model_4():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(400, tf.nn.relu))
    model.add(tf.keras.layers.Dense(10, tf.nn.relu))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))
    
    return model

def create_model_5():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(600, tf.nn.relu))
    model.add(tf.keras.layers.Dense(10, tf.nn.relu))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))
    
    return model

def create_model_6():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(800, tf.nn.relu))
    model.add(tf.keras.layers.Dense(10, tf.nn.relu))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))
    
    return model