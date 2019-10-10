import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


from keras.models import Sequential
from keras.layers import Input,Dense, Embedding, LSTM, Bidirectional, GlobalAveragePooling1D, TimeDistributed, GlobalMaxPooling1D
from keras.initializers import Constant


from keras.utils import to_categorical
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

import keras.backend as K
from gensim.models import KeyedVectors 

#from AttentionLayers import *

from keras.optimizers import SGD, Adam, Adadelta, Adagrad, RMSprop, Adamax, Nadam

from keras.callbacks import ReduceLROnPlateau

from keras.regularizers import l2

from keras.models import load_model

import keras


abspath = '/home/pnguyen/projects/DAProject'

EMBEDDING_DIM = 300 # how big is each word vector
VOCAB_SIZE = 40000 # how many unique words to use (i.e num rows in embedding vector)
MAX_LENGTH = 100 # max number of words in a comment to use
VALIDATION_SPLIT = 0.3

num_classes = 42

#######################################################
def load_data(path, verbose=True):
    with open(path, 'rb') as file:
        saved_data = pickle.load(file)
        file.close()
    if verbose:
        print("Loaded data from file %s." % path)
    return saved_data

########################################################
def encode(df):
    index = df.index
    tokenizer_obj = Tokenizer(num_words = VOCAB_SIZE, filters='')  ## ,filters = '([+/\}\[\]]|\{\w),'
    tokenizer_obj.fit_on_texts(list(df['text']))
    word_index = tokenizer_obj.word_index
    sequences_train = tokenizer_obj.texts_to_sequences(df['text'])
    #MAX_LENGTH = max([len(sequence) for sequence in sequences_train])
    X = pad_sequences(sequences_train, maxlen=MAX_LENGTH )
    dfX = pd.DataFrame(X, index = index)
    return dfX, word_index

######## EMBEDDING MATRIX #######################################
'''
def create_embedding_matrix(path, word_index):  
    #word_index = tokenizer_obj.word_index
    nb_words = min(VOCAB_SIZE, len(word_index))+1
    embedding_index = KeyedVectors.load_word2vec_format(glove_path, binary=True) 
  
    embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
    for word, i in word_index.items():
        if i >= VOCAB_SIZE: 
            continue
        if word in embedding_index:
            #words not found in embedding index will be all-zeros
            embedding_vector = embedding_index.get_vector(word)
            embedding_matrix[i] = embedding_vector
    
    return embedding_matrix
'''

def create_embedding_matrix(glove_path, word_index):   #'/home/phuong/Desktop/glove.6B/glove.6B.300d.txt'
    f = open(glove_path, "r", encoding="utf8")
    #word_index = tokenizer_obj.word_index
    nb_words = min(VOCAB_SIZE, len(word_index))+1
    #nb_words = 2000
    #Get embeddings from Glove/home/phuong/PHUONG/Code/optimized_code/glove.6B
    embedding_index = dict()
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embedding_index[word] = coefs
    f.close()
    embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
    for word, i in word_index.items():
        if i >= VOCAB_SIZE: 
            continue
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None: 
            #words not found in embedding index will be all-zeros
            embedding_matrix[i] = embedding_vector
    return embedding_matrix


############ define model ##############################
def Model_RNN_LSTM(embedding_matrix, num_classes):# trainable = True
    model = Sequential()
    embedding_layer = Embedding(VOCAB_SIZE,
                            EMBEDDING_DIM,
                            embeddings_initializer= Constant(embedding_matrix),
                            input_length=MAX_LENGTH,
                            trainable=False)
    '''
    embedding_layer = Embedding(VOCAB_SIZE,
                            EMBEDDING_DIM,
                            embeddings_initializer =  Constant(embedding_matrix),
                            input_length=MAX_LENGTH,
                            mask_zero=False)
    '''
    model.add(embedding_layer)
    #model.add(LSTM(units=128, dropout=0.3,kernel_initializer='random_uniform', recurrent_initializer='glorot_uniform', return_sequences=True))       
    #model.add(Bidirectional(LSTM(units=32, dropout=0.2, recurrent_dropout=0.2, return_sequences=True,kernel_initializer='random_uniform', recurrent_initializer='glorot_uniform')))  
    #model.add(Bidirectional(LSTM(units=32, dropout=0.2, recurrent_dropout=0.2, return_sequences=True,kernel_initializer='random_uniform', recurrent_initializer='glorot_uniform')))  
    model.add(Bidirectional(LSTM(units=128, dropout=0.5, recurrent_dropout=0.5, return_sequences=True)))
    model.add(Bidirectional(LSTM(units=128, dropout=0.5, recurrent_dropout=0.5)))    
    #model.add(TimeDistributed(Dense(128, input_shape=(MAX_LENGTH, 128))))
    #model.add(GlobalMaxPooling1D())
    model.add(Dense(num_classes, activation='softmax'))
    # try using different optimizers and different optimize cofigs
    adam = Adam(lr= 1e-3)
    #rmsprop= RMSprop(lr=1e-3, decay=0.001)
    #sgd = SGD(lr = 1e-3, decay = 0.7)
    model.compile(loss='categorical_crossentropy', optimizer= adam, metrics=['accuracy'])
    print(model.summary())
    return model

####################################
LAYER_DESIRED = 2


def genData(df, X, model):
    get_layer_output = K.function([model.layers[0].input, K.learning_phase()],
                                   [model.layers[LAYER_DESIRED].output])
    dfdlnum = df['conversation_no'].tolist()
    index = df.index
    X = pd.DataFrame(X, index = index)
    dfnum = []
    for i in dfdlnum:
        if i not in dfnum:
            dfnum.append(i)
    
    Xbatch =  []
    Ybatch = []
    for i in dfnum:
        Xi = X.loc[df['conversation_no']== i].values         #.astype(np.float64)
        Xi.astype(np.float64)
        dfTemp = df.loc[df['conversation_no']== i]
        Yi = dfTemp['label'].values.astype(np.int64) + 1
        Xbatchi = get_layer_output([Xi, 0])[0]
        Xbatch.append(Xbatchi)
        Ybatch.append(Yi)
    Xmul = np.concatenate([fi for fi in Xbatch], axis = 0)
    Ymul = df['label'].values 
    return Xmul, Ymul, Xbatch, Ybatch
######################################
path_data = "../data/"
path_glove = "../embedding/"
path_ouput = "../output/"



df_data = load_data(path_data+'swda.pkl')

MAX_LENGTH = max([len(x.split()) for x in list(df_data['text']) ]) 
dfX, word_index = encode(df_data)
 
df_data['label'] = df_data['label'].astype('category')
Y_data = to_categorical(df_data['label'], num_classes = num_classes)

embedding_matrix = create_embedding_matrix(path_glove +'glove.6B.300d.txt', word_index)

from keras.callbacks import EarlyStopping, ModelCheckpoint

Kfold = 10
acLSTMtest = []
for i in range(Kfold):
    
    idx = list(set(df_data['conversation_no']))
    
    train_idx, test_idx  = train_test_split(idx, test_size=0.3)
    
    df_train = df_data.loc[df_data['conversation_no'].isin(train_idx)]
    df_test = df_data.loc[df_data['conversation_no'].isin(test_idx)]
    
    X_train = dfX.loc[df_data['conversation_no'].isin(train_idx)].values
    X_test = dfX.loc[df_data['conversation_no'].isin(test_idx)].values
    
    
    Ymul_train =  df_train['label'].values
    Ymul_test =  df_test['label'].values
    
    Y_train = to_categorical(Ymul_train, num_classes = num_classes)
    Y_test = to_categorical(Ymul_test, num_classes =  num_classes)
    
    
    #X_train, X_test, Y_train, Y_test = train_test_split(dfX, Y_data, test_size=0.3, random_state=42)
    
    
    
    model = Model_RNN_LSTM(embedding_matrix, num_classes)    

    # Set callback functions to early stop training and save the best model so far
    callbacks = [EarlyStopping(monitor='val_acc', patience=5, verbose = 1),
                 ModelCheckpoint(filepath= path_ouput+ 'model_'+ str(i)+'.h5', monitor='val_acc', save_best_only=True, verbose = 1)]
    
     
    history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test)
                              , epochs=30, batch_size= 32, verbose=1, callbacks=callbacks)
    
    
    #abspath2 = '/home/pnguyen/projects/DAProject/FEATURE'
    
    #model.save(abspath2+ 'model_'+ str(i)+'.h5')  # creates a HDF5 file 'my_model.h5'    
    
    model = load_model(path_ouput + 'model_'+ str(i)+'.h5')
    
    # Test set
    test_scores = model.evaluate(X_test, Y_test, verbose=2)
    print("Test data: ")
    print("Loss: ", test_scores[0], " Accuracy: ", test_scores[1])
    
    acLSTMtest.append(test_scores[1])

    Xmul_train, Ymul_train, Xbatch_train, Ybatch_train = genData(df_train, X_train, model)
    Xmul_test, Ymul_test, Xbatch_test, Ybatch_test = genData(df_test, X_test, model)
    
    ########
    np.save(path_ouput + 'X_train'+str(i), X_train)
    np.save(path_ouput + 'X_test'+str(i), X_test)
    
    np.save(path_ouput + 'Y_train'+str(i), Y_train)
    np.save(path_ouput + 'Y_test'+str(i), Y_test)
    
    np.save(path_ouput + 'Xmul_train'+str(i), Xmul_train)
    np.save(path_ouput + 'Xmul_test'+str(i), Xmul_test)    
    
    np.save(path_ouput + 'Ymul_train'+str(i), Ymul_train)
    np.save(path_ouput + 'Ymul_test'+str(i), Ymul_test)

    
    np.save(path_ouput + 'Xbatch_train'+str(i), Xbatch_train)
    np.save(path_ouput + 'Xbatch_test'+str(i), Xbatch_test)
    
    np.save(path_ouput + 'Ybatch_train'+str(i), Ybatch_train)
    np.save(path_ouput + 'Ybatch_test'+str(i), Ybatch_test)
    
print(acLSTMtest)
text_file = open(path_ouput + 'accuracy_LSTM.txt', 'w')
print("Test data:   : {}".format(acLSTMtest), file=text_file)
text_file.close()