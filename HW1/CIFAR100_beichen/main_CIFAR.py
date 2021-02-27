
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
import util_CIFAR as util
import model_CIFAR as model
import tensorflow_datasets as tfds
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam

#loading training, validation, and test data sets
X_train, y_train_encoded, X_val, y_val_encoded, X_test, y_test = util.loading_cifar100()

print('The size of training data: ',X_train.shape[0])

#image augmentation
datagen = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=45, width_shift_range = 0.2, height_shift_range = 0.2, horizontal_flip= True, fill_mode='nearest')

#define model
ResNet = model.model_def()
ResNet.build(input_shape=(40000, 32, 32, 3))
ResNet.summary()


#train model
checkpoint_path = "./training_checkpoints"
es_callback =EarlyStopping(monitor='val_accuracy', min_delta=0.0001, patience = 10)
cp_callback = ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=0, save_freq='epoch')
training = ResNet.fit(datagen(X_train,y_train_encoded, batch_size = 400), epochs = 100, callbacks = [es_callback,cp_callback],validation_data = (X_val,y_val_encoded), verbose = 1)
_, accuracy = ResNet.evaluate(X_val, y_val_encoded, verbose=0)
print('Accuracy on the validation set: %.3f' % (accuracy * 100.0))

#plot learning curves
util.plot_diagnostics(training)
#save the model
tf.saved_model.save(ResNet, 'saved_model_test')

#prediction
predict =  ResNet.predict(X_test, verbose=1)
pred_label = tf.argmax(predict, axis = 1)
pred_accu = (np.sum(pred_label==y_test)/len(pred_label))*100
print('Accuracy on the test set: %.3f' % (pred_accu))
confusion_matrix = tf.math.confusion_matrix(labels=y_test, predictions=pred_label).numpy()
print(confusion_matrix)
np.savetxt("confusion_matrix.txt", confusion_matrix)


#confidence interval
n = X_test.shape[0]
low, high = util.confidence_interval(pred_accu/100,n)
print("95%% confidence interval:[%.1f%%, %.1f%%]:" % (low*100, high*100))
