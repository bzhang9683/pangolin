from sklearn.model_selection import StratifiedKFold
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
import nltk
import util
import model
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam


#loading training, validation, and test data sets
X_train, y_train, X_test, y_test = util.loading_imdb()

print('The size of training data: ',X_train.shape[0])

#cleaning the data set
X_train_preprocessed = util.preprocessing_text(X_train.numpy())
X_test_preprocessed = util.preprocessing_text(X_test.numpy())

#tokenize data
X_train_seq, X_test_seq, vocab_size, word_index_items = util.to_seuqnce_data(X_train_preprocessed, X_test_preprocessed, max_length = 64)

#load pretrained embedding
embeddings_index = util.load_pretrained_embedding()

print('Loaded %s word vectors.' % len(embeddings_index))


#train model
checkpoint_path = "./training_checkpoints"
es_callback =EarlyStopping(monitor='val_accuracy', min_delta=0.0001, patience = 5)
cp_callback = ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=0, save_freq='epoch')


kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cvscores = []
for train, test in kfold.split(X_train_seq, y_train):
    LSTM = model.def_model(vocab_size = vocab_size, word_index = word_index_items, embeddings_index = embeddings_index, pretrain_embedding = True, global_pooling = True)
    history = LSTM.fit(tf.gather(X_train_seq,train), tf.gather(y_train,train), batch_size=256, epochs=30, validation_split=0.2,
                    callbacks=[es_callback,cp_callback],verbose=1,shuffle=True)
    # evaluate the model
    scores = LSTM.evaluate(tf.gather(X_train_seq,test), tf.gather(y_train,test), verbose=0)
    print("%s: %.2f%%" % (LSTM.metrics_names[1], scores[1]*100))
    cvscores.append(scores[1] * 100)
    
print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))

#save model
tf.saved_model.save(LSTM, 'saved_model_best')


#plot learning curves
util.plot_diagnostics(history)

#prediction
predict =  LSTM.predict(X_test_seq)
predict[predict>=0.5] = 1
predict[predict<0.5] = 0
label = predict
pred_accu = np.sum(label==y_test.numpy().reshape(5000,1))/len(label) 
print('Accuracy on the test set: %.3f%%' % (pred_accu*100))

#calculate and plot confusion matrix
confusion_matrix = tf.math.confusion_matrix(labels=y_test, predictions=label).numpy()
print(confusion_matrix)
np.savetxt("confusion_matrix.txt", confusion_matrix)

util.plot_CM(confusion_matrix)

#confidence interval
n = X_test.shape[0]
low, high = util.confidence_interval(pred_accu,n)
print("95%% confidence interval:[%.1f%%, %.1f%%]:" % (low*100, high*100))



