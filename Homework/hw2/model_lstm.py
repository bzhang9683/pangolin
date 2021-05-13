#build LSTM model
import tensorflow as tf
import numpy as np
import nltk
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import LayerNormalization
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import GlobalMaxPooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model
from tensorflow.keras.layers import Input


def def_model(vocab_size, word_index, embeddings_index,vector_size = 300, pretrain_embedding = True, global_pooling = True):
    words = Input(shape=(None,))
    # We'll make a conv layer to produce the query and value tensors
    query_layer = tf.keras.layers.Conv1D(
        filters=100,
        kernel_size=4,
        padding='same')
    value_layer = tf.keras.layers.Conv1D(
        filters=100,
        kernel_size=4,
        padding='same')
    # Then they will be input to the Attention layer
    attention = tf.keras.layers.Attention()
    concat = tf.keras.layers.Concatenate()
    
    if pretrain_embedding:
        # create a weight matrix for words in training docs
        embedding_matrix = np.zeros((vocab_size, vector_size))
        for word, i in word_index:
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
        embeddings = Embedding(vocab_size, vector_size, weights=[embedding_matrix], trainable=False)(words)
    else:
        embeddings = Embedding(vocab_size,vector_size,trainable=True)(words)
    query = query_layer(embeddings)
    value = value_layer(embeddings)
    query_value_attention = attention([query, value])
    attended_values = concat([query, query_value_attention])
    X = LSTM(100, activation='tanh', return_sequences=True)(attended_values)
    X = LSTM(100, activation='tanh', return_sequences=True)(X)
    if global_pooling:
        X = LSTM(100, activation='tanh', return_sequences=True)(X)
        X = GlobalMaxPooling1D()(X)
    else:
        X = LSTM(100, activation='tanh')(X)
    X = Dense(128,activation='relu')(X)
    #X = Dense(64,activation='relu')(X)
    X = Dropout(0.3)(X)
    results = Dense(1,activation='sigmoid')(X)
    opt = tf.keras.optimizers.Adam(learning_rate=0.01)
    model = Model(inputs=words, outputs=[results])
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    
    return model
