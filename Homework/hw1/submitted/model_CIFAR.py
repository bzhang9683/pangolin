import tensorflow as tf
import numpy
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, ZeroPadding2D, BatchNormalization

# Create the model
def model_create(no_classes,input_shape):
	model = Sequential()
	model.add(ZeroPadding2D(4, input_shape = input_shape))
# stack 1
	model.add(Conv2D(40, kernel_size=(3, 3), padding = 'same', activation='relu', input_shape=input_shape, kernel_regularizer=tf.keras.regularizers.l2(0.01)))
	model.add(BatchNormalization())
	model.add(Conv2D(80, kernel_size=(3, 3), padding = 'same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(BatchNormalization())
	model.add(Dropout(.2))
#stack 2
	model.add(Conv2D(160, kernel_size=(3, 3), padding = 'same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
	model.add(BatchNormalization())
	model.add(Conv2D(320, kernel_size=(3, 3), padding = 'same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(BatchNormalization())
	model.add(Dropout(.2))
#stack 3
	model.add(Conv2D(640, kernel_size=(3, 3), padding = 'same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
	model.add(BatchNormalization())
	model.add(Conv2D(1280, kernel_size=(3, 3), padding = 'same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(BatchNormalization())
	model.add(Dropout(.2))
#stack flat	
	model.add(Flatten())
	model.add(Dense(1024, activation='relu'))
	#model.add(Dense(128, activation='relu'))
	model.add(Dense(600, activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.01)))
#	model.add(Dense(128, activation='relu',kernel_regularizer=tf.keras.regularizers.l1(0.01), activity_regularizer=tf.keras.regularizers.l2(0.01)))
	model.add(Dense(no_classes, activation='softmax'))
	return model
