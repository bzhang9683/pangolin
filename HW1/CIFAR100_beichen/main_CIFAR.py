#%%
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import util_CIFAR as util
import model_CIFAR as model
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
#%%
#loading training, validation, and test data sets
X_train, y_train_encoded, X_val, y_val_encoded, X_test, y_test = util.loading_cifar100()

print('The size of training data: ',X_train.shape[0])
#%%
#define model
ResNet = model.model_def()
ResNet.build(input_shape=(30000, 32, 32, 3))
ResNet.summary()

#%%
#train model
es_callback =EarlyStopping(monitor='val_accuracy', min_delta=0.0001, patience = 1)
training = ResNet.fit(X_train,y_train_encoded, epochs = 50, batch_size = 300, callbacks = [es_callback],validation_data = (X_val,y_val_encoded), verbose = 1)
_, accuracy = ResNet.evaluate(X_val, y_val_encoded, verbose=0)
print('Accuracy on the validation set: %.3f' % (accuracy * 100.0))
#%%
#plot learning curves
util.plot_diagnostics(training)
#%%
#prediction
predict =  ResNet.predict(X_test, verbose=1)
pred_label = tf.argmax(predict, axis = 1)
pred_accu = (np.sum(pred_label==y_test)/len(pred_label))*100
print('Accuracy on the test set: %.3f' % (pred_accu))
confusion_matrix = tf.math.confusion_matrix(labels=y_test, predictions=pred_label).numpy()
print(confusion_matrix)
np.savetxt("confusion_matrix.txt", confusion_matrix)

# %%
