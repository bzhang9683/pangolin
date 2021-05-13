import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np
import re
from bs4 import BeautifulSoup
import unicodedata
from nltk.tokenize.toktok import ToktokTokenizer
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.text import Tokenizer
from tqdm import tqdm
import codecs
import nltk
import itertools
def loading_cifar100():
    '''
    Required package: TensorFlow
    '''
def loading_imdb():
    '''
    Required package: TensorFlow
    '''
    (train_ds,test_ds),meta_info = tfds.load('imdb_reviews/plain_text',data_dir='/work/tadesse/beichen/Homework_CSCE879/hw1/',with_info=True, split=['train[:80%]','train[80%:]'], batch_size=-1)
    # If batch_size=-1, will return the full dataset as tf.Tensors.
    X_train = train_ds['text']
    y_train = train_ds['label']
    X_test = test_ds['text']
    y_test = test_ds['label']
    
    return X_train, y_train, X_test, y_test

def plot_diagnostics(history):
    #plot loss curve
    plt.figure()
    plt.plot(history.history['loss'], color='blue', label='train')
    plt.plot(history.history['val_loss'], color='red', label='validation')
    plt.xlabel('Epoch')
    plt.legend()
    plt.title('Cross Entropy Loss')
    plt.savefig('loss_curve.png')
    #plot accuracy curve
    plt.figure()
    plt.plot(history.history['accuracy'], color='blue', label='train')
    plt.plot(history.history['val_accuracy'], color='red', label='validation')
    plt.xlabel('Epoch')
    plt.legend()
    plt.title('Classification Accuracy')
    plt.savefig('accu_curve.png')

def confidence_interval(error,n):
    low = error -1.96 *np.sqrt((error * (1-error)) / n)
    high = error + 1.96 *np.sqrt((error * (1-error)) / n)
    return low, high
#remove html tags
def remove_html_tags(text):
    soup = BeautifulSoup(text, "html.parser")
    [s.extract() for s in soup(['iframe','script'])]
    stripped_text = soup.get_text()
    stripped_text = re.sub(r'["\|\n|\r|\n\r]+','', stripped_text)
    return stripped_text

#removing accented characters
def remove_accented_chars(text):
    text = unicodedata.normalize('NFKD', text).encode('ascii','ignore').decode('utf-8','ignore')
    return text

#removing special characters:
def remove_special_characters(text):
    
    text = re.sub(r'[^a-zA-z\s]','',text)
    return text

#removing stopwords

def remove_stopwords(text):
    
    tokenizer = ToktokTokenizer()
    stopword_list = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", 
                 "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", 
                 "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", 
                 "their", "theirs", "themselves", "what", "which", "who", "whom", "this", 
                 "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", 
                 "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", 
                 "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", 
                 "of", "at", "by", "for", "with", "about", "against", "between", "into", 
                 "through", "during", "before", "after", "above", "below", "to", "from", 
                 "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", 
                 "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", 
                 "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", 
                 "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", 
                 "will", "just", "don", "should", "now"]   
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    filtered_tokens = [token for token in tokens if token not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)    
    
    return filtered_text

def preprocessing_text(text_arr):
    preprocessed_text = []
    idx = 0
    for text in text_arr:
        text = remove_html_tags(text) #remove html tags
        text = remove_accented_chars(text) #removing accented characters
        text = remove_special_characters(text) #removing special characters
        text = text.lower() #change to lower case
        text = remove_stopwords(text) #removing stopwords
        #word_seq,vocab_size = toakenizing(text)
        #max_vocab_size = np.max(vocab_size)
        preprocessed_text.append(text)
        idx+=1
    print('Data Preprocessing finished.')
    return preprocessed_text

def to_seuqnce_data(X_train, X_test, max_length):
    tokenizer = Tokenizer()
    X = X_train + X_test
    tokenizer.fit_on_texts(X)
    word_index_items = tokenizer.word_index.items()
    vocab_size = len(tokenizer.word_index) + 1
    word_seq_train = tokenizer.texts_to_sequences(X_train)
    word_seq_test = tokenizer.texts_to_sequences(X_test)
    X_train_seq = sequence.pad_sequences(word_seq_train, maxlen=max_length)
    X_test_seq = sequence.pad_sequences(word_seq_test, maxlen=max_length)
    
    return X_train_seq, X_test_seq, vocab_size, word_index_items
def load_pretrained_embedding(fdir = "/work/tadesse/beichen/Homework_CSCE879/hw2/glove.840B.300d.txt"):
    embeddings_index = {}
    f = codecs.open(fdir, encoding="utf-8")
    for line in tqdm(f):
        values = line.rstrip().rsplit(' ')
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('Done!')
    return embeddings_index
def plot_CM(confusion_matrix):
    plt.imshow(confusion_matrix,cmap=plt.cm.Blues,interpolation='nearest')
    plt.colorbar()
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    tick_marks = np.arange(2) # length of classes
    class_labels = ['0','1']
    tick_marks
    plt.xticks(tick_marks,class_labels)
    plt.yticks(tick_marks,class_labels)
    # plotting text value inside cells
    thresh = confusion_matrix.max() / 2.
    for i,j in itertools.product(range(confusion_matrix.shape[0]),range(confusion_matrix.shape[1])):
        plt.text(j,i,format(confusion_matrix[i,j],'d'),horizontalalignment='center',color='white' if confusion_matrix[i,j] >thresh else 'black')
    plt.savefig('CFM_plot.png')
