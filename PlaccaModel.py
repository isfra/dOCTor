import numpy as np
from keras import layers
from keras import regularizers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, Dropout, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
#import pydot                                                         non so che contiene!!!!!!!!!!
#from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
#from resnets_utils import *
from keras.initializers import glorot_uniform
import scipy.misc
from matplotlib.pyplot import imshow
#matplotlib inline
import keras.backend as K

K.set_image_data_format('channels_last')
K.set_learning_phase(1)

l2_lambda=0
drop_prob=0


"""
Implementation of the identity block 

le immagini rimangono delle stesse dim dopo il passaggio. il numero di canali in genere aumenta (F3 è il finale)

E importante che X abbia le stesse dim di X_shortcut che viene aggiunto dopo i filtraggi senza modifiche

Arguments:
X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
f_h f_l -- integers, specifying the shape of the middle CONV's window for themain path
filters -- python list of integers, defining the number of filters in the CONV layers of the main path
stage -- integer, used to name the layers, depending on their position in the network
block -- string/character, used to name the layers, depending on their position in the network
Returns:
X -- output of the identity block, tensor of shape (n_H, n_W, n_C) 
"""
def identity_block(X, f_l, filters, stage, block, f_h=1):
    # defining name basis
    conv_name_base = 'res-id' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    # Retrieve Filters
    F1, F2, F3 = filters
    # Save the input value. You'll need this later to add back to the mainpath.
    X_shortcut =X
    '''     padding same = fare si che l output abbia le stesse dim dell input aggiungendo zeri prima
            padding valid non agginge zeri ma se kernel_size = (1, 1) e strides = (1,1) la dim non cambia (come qua ma in convblock no)'''
    # First component of main path
    X = Conv2D(filters = F1, kernel_size = (1, 1), strides = (1,1), padding= 'valid',
               name = conv_name_base + '2a-' + str(F1) + 'Filters' + str(1)+ 'x' + str(1) + '-strides' + str(1) + 'x' +str(1)+'-valid', 
               kernel_initializer =glorot_uniform(seed=0),  kernel_regularizer=regularizers.l2(l2_lambda))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)
    # Second component of main path (≈3 lines)
    X = Conv2D(filters = F2, kernel_size = (f_h,f_l), strides = (1,1), padding= 'same', 
               name = conv_name_base + '2b-'+ str(F1) + 'Filters' + str(f_h)+ 'x' + str(f_l) + '-strides' + str(1) + 'x' +str(1)+'-same', 
               kernel_initializer = glorot_uniform(seed=0),  kernel_regularizer=regularizers.l2(l2_lambda))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)
    # Third component of main path (≈2 lines)
    X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1,1), padding= 'valid', 
               name = conv_name_base + '2c-'+ str(F1) + 'Filters' + str(1)+ 'x' + str(1) + '-strides' + str(1) + 'x' +str(1)+'-valid', 
               kernel_initializer =glorot_uniform(seed=0),  kernel_regularizer=regularizers.l2(l2_lambda))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)
    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = layers.add([X, X_shortcut])
    X = Activation('relu')(X)
    return X




"""
Implementation of the convolutional block 

le immagini rimangono delle stesse dim dopo il passaggio le dimensioni si accorciano rispetto a s__h s_l e  f_hconv, f_lconv.
(gli f ) il numero di canali in genere aumenta (F3 è il finale)

E importante che X abbia le stesse dim di X_shortcut che viene aggiunto dopo i filtraggi 
con un layer di convoluzione indipendente dagli altri 3 che ne riduce la dimensione

Arguments:
X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
 f_hconv, f_lconv --- integers, specifying the shape of the first and shortcut CONV's window for the main path
f_h,f_l -- integers, specifying the shape of the middle CONV's window for the main path
filters -- python list of integers, defining the number of filters in the CONV layers of the main path
stage -- integer, used to name the layers, depending on their position in the network
block -- string/character, used to name the layers, depending on their position in the network
s_h s_l -- Integers, specifying the stride to be used
Returns:
X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
"""

def convolutional_block(X, f_hconv, f_lconv, f_h, f_l, filters, stage, block, s_h = 2, s_l=1):
    # defining name basis
    conv_name_base = 'res-conv' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    # Retrieve Filters
    F1, F2, F3 = filters
    # Save the input value
    X_shortcut = X
    # First component of main path
    X = Conv2D(F1, (f_hconv, f_lconv), strides = (s_h,s_l), padding = 'valid', 
               name = conv_name_base +'2a-'+ str(F1) + 'Filters' + str(f_hconv)+ 'x' + str(f_lconv) + '-strides' + str(s_h) + 'x' +str(s_l)+'-valid',
               kernel_initializer = glorot_uniform(seed=0), kernel_regularizer=regularizers.l2(l2_lambda))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)
    # Second component of main path (≈3 lines)
    X =Conv2D(F2, (f_h, f_l), strides = (1,1), padding = 'same', 
              name = conv_name_base +'2b-'+ str(F2) + 'Filters' + str(f_h)+ 'x' + str(f_l) + '-strides' + str(1) + 'x' +str(1) + 'same', 
              kernel_initializer = glorot_uniform(seed=0),  kernel_regularizer=regularizers.l2(l2_lambda))(X) # f,2
    BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    Activation('relu')(X)
    # Third component of main path (≈2 lines)
    X = Conv2D(F3, (1, 1), strides = (1,1), padding = 'valid', 
               name = conv_name_base +'2c-' + str(F3) + 'Filters' + str(1)+ 'x' + str(1) + '-strides' + str(1) + 'x' +str(1)+'-valid',
                kernel_initializer = glorot_uniform(seed=0),  kernel_regularizer=regularizers.l2(l2_lambda))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)
    #shotcut path
    X_shortcut = Conv2D(F3, (f_hconv, f_lconv), strides = (s_h,s_l), padding = 'valid', 
                        name= conv_name_base + '1-' + str(F3) + 'Filters' + str(f_hconv)+ 'x' + str(f_lconv) + '-strides' + str(s_h) + 'x' +str(s_l)+'-valid', 
                        kernel_initializer = glorot_uniform(seed=0),  kernel_regularizer=regularizers.l2(l2_lambda))(X_shortcut)
    X_shortcut = BatchNormalization(axis = 3, name = bn_name_base + '1')(X_shortcut)
    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = layers.add([X ,X_shortcut])
    X = Activation('relu')(X)
    return X


"""

Implementation of the popular ResNet50 the following architecture:

La funzione crea un modello Keras, che crea un grafo di calcolo di tensorflowm ponendo una variabile = funz dichiariamo una classe Modello di Keras
 che ha tutte funzionalità di calcolo dei gradienti analisi ets.... vedi Train.py

Qui chiamiamo i blocchi con i rispettivi iperparametri. ogni blocco corrisponde a 3 livelli con uno sdoppiamento del filtraggio 
(identity è nature, in conv anche la scorcatoia viene convoluta prima di essere sommata al momento del calcolo delle attivazioni dell output)  vedi sopra

CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CON
VBLOCK -> IDBLOCK*3
-> CONVBLOCK -> IDBLOCK*5 -> ///saltato CONVBLOCK -> IDBLOCK*2 -> AVGPOOL////// -> TOPLAYER
Arguments:
input_shape -- shape of the images of the dataset
classes -- integer, number of classes
Returns:
model -- a Model() instance in Keras
"""

def ResNet50(input_shape, reg, classes = 2):
    
    l2_lambda=reg
    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)
    # Zero-Padding
    X = ZeroPadding2D((0, 3))(X_input) # 3,3
    # Stage 1
    X = Conv2D(64, (1, 7), strides = (1, 2), name = 'conv1-'+ str(64) + 'Filters' + str(1)+ 'x' + str(7) + '-strides' + str(1) + 'x' +str(2)+'-valid',
               kernel_initializer = glorot_uniform(seed=0),  kernel_regularizer=regularizers.l2(l2_lambda))(X_input)
    X = BatchNormalization(axis = 3, name = 'bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((1,3), strides=(1, 2))(X)
    # Stage 2
    X = convolutional_block(X, f_hconv=1, f_lconv=6, f_h=1,f_l = 3, filters = [64, 64, 256], stage = 2, block='a', s_h=1, s_l = 1)
    X = identity_block(X, 5, [64, 64, 256], stage=2, block='b')
    X = identity_block(X, 5, [64, 64, 256], stage=2, block='c')
    # Stage 3 (≈4 lines)
    X = convolutional_block(X, f_hconv=2, f_lconv=7, f_h=1, f_l = 3, filters = [128,128,512], stage = 3, block='a', s_h=1 , s_l = 3)
    X = identity_block(X, 4, [128,128,512], stage=3, block='b')
    X = identity_block(X, 4, [128,128,512], stage=3, block='c')
    X = identity_block(X, 3, [128,128,512], stage=3, block='d')
    # Stage 3 (≈4 lines)   questo era di troppo e ce l ho messso
    X = convolutional_block(X, f_hconv=2, f_lconv=5, f_h=1, f_l = 3, filters = [128,128,512], stage = 4, block='a', s_h=1 , s_l = 2)
    X = identity_block(X, 3, [128,128,512], stage=4, block='b')
    X = identity_block(X, 3, [128,128,512], stage=4, block='c')
    X = identity_block(X, 3, [128,128,512], stage=4, block='d')
    ''''X = identity_block(X, 3, [128,128,512], stage=4, block='c')
    X = identity_block(X, 3, [128,128,512], stage=4, block='d')
    # Stage 4 (≈6 lines)
    X = convolutional_block(X,  f_hconv=1, f_lconv=2,f_h=1, f_l=3, filters=[256, 256, 1024], stage = 5, block='a', s_h=1, s_l = 2)
    X = identity_block(X, 3, [256, 256, 1024], stage=5, block='b')
    X = identity_block(X, 3, [256, 256, 1024], stage=5, block='c')
    X = identity_block(X, 3, [256, 256, 1024], stage=5, block='d')
    X = identity_block(X, 3, [256, 256, 1024], stage=5, block='e')
    X = identity_block(X, 3, [256, 256, 1024], stage=5, block='f')
    
    # Stage 5 (≈3 lines)
    #X = convolutional_block(X, f_hconv=1, f_lconv=2, f_l = 3, f_h=1, filters = [512, 512, 2048], stage = 5,block='a', s_l = 2, s_h=1)
    #X = identity_block(X, 3, [512, 512, 2048], stage=5, block='b')
    #X = identity_block(X, 3, [512, 512, 2048], stage=5, block='c')'''
    # AVGPOOL (≈1 line). Use "X = AveragePooling2D(...)(X)"
    X = AveragePooling2D(pool_size=(1, 3), strides=None, padding='valid',name='avg_pool')(X)
    '''  qua appiattiamo il network in un vettore per finire classico fully connected con sigmoid   '''
    # output layer
    X = Flatten()(X)
    if (classes==2):
        X = Dense(1, activation='sigmoid', name='fc-sigmoid')(X)
    else:
        X = Dense(classes, activation='softmax', name='fc-softmax' + str(classes),kernel_initializer = glorot_uniform(seed=0))(X)
    # Create model
    """"    anche se ancora non abbiamo passato i dati da adesso in poi il modello sa quello che deve fare. tipo costruttore 
            verranno poi inseriti i dati e fatti i calcoli tramite delle sue funzionalità """
            
    model = Model(inputs = X_input, outputs = X, name='ResNet50')
    
    return model

def modellino_prova(input_shape , reg, classes = 2):
    X_input = Input(input_shape)
    
    X = Conv2D(64, (3, 100), strides = (1, 50), name = 'conv1',kernel_initializer = glorot_uniform(seed=0))(X_input)
    
    X = AveragePooling2D(pool_size=(2, 3), strides=None, padding='valid',name='avg_pool')(X)
        # output layer
    X = Flatten()(X)
    if (classes==2):
        X = Dense(1, activation='sigmoid', name='fc-sigmoid')(X)
    else:
        X = Dense(classes, activation='softmax', name='fc-softmax' + str(classes),kernel_initializer = glorot_uniform(seed=0))(X)
    # Create model           
    model = Model(inputs = X_input, outputs = X, name='solo_conv')

    return model
   

def modellino(input_shape , reg, classes = 2):

    l2_lambda=reg
    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)
       # Stage 1
    X = Conv2D(64, (1, 7), strides = (1, 2), name = 'conv1',kernel_initializer = glorot_uniform(seed=0))(X_input)
    X = BatchNormalization(axis = 3, name = 'bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((1,4), strides=(1, 2))(X)
    # Stage 2
    X = convolutional_block(X, f_hconv=1, f_lconv=6, f_h=1,f_l = 3, filters = [64, 64, 256], stage = 2, block='a', s_h=1, s_l = 1)
    # Stage 3 (≈4 lines)
    X = convolutional_block(X, f_hconv=1, f_lconv=7, f_h=1, f_l = 3, filters = [128,128,512], stage = 3, block='a', s_h=1 , s_l = 3)
    # Stage 4 (≈6 lines)
    X = convolutional_block(X,  f_hconv=2, f_lconv=5,f_h=1, f_l=3, filters=[256, 256, 1024], stage = 4, block='a', s_h=1, s_l = 2)
    # Stage 5 (≈3 lines)
    X = convolutional_block(X, f_hconv=1, f_lconv=5, f_h=1, f_l = 3, filters = [512, 512, 2048], stage = 5,block='a', s_h=1, s_l = 2)
    # AVGPOOL (≈1 line). Use "X = AveragePooling2D(...)(X)"
    X = AveragePooling2D(pool_size=(1, 3), strides=None, padding='valid',name='avg_pool')(X)
        # output layer
    X = Flatten()(X)
    if (classes==2):
        X = Dense(1, activation='sigmoid', name='fc-sigmoid')(X)
    else:
        X = Dense(classes, activation='softmax', name='fc-softmax' + str(classes),kernel_initializer = glorot_uniform(seed=0))(X)
    # Create model           
    model = Model(inputs = X_input, outputs = X, name='solo_conv')

    return model

def modellino_bis(input_shape , reg, classes = 2):

    l2_lambda=reg
    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)
       # Stage 1
    X = Conv2D(64, (1, 7), strides = (1, 2), name = 'conv1',kernel_initializer = glorot_uniform(seed=0))(X_input)
    X = BatchNormalization(axis = 3, name = 'bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((1,4), strides=(1, 2))(X)
    # Stage 2
    X = convolutional_block(X, f_hconv=2, f_lconv=7, f_h=1,f_l = 3, filters = [64, 64, 256], stage = 2, block='a', s_h=1, s_l = 1)
    # Stage 3 (≈4 lines)
    X = convolutional_block(X, f_hconv=2, f_lconv=7, f_h=1, f_l = 3, filters = [128,128,512], stage = 3, block='a', s_h=1 , s_l = 3)
    # Stage 4 (≈6 lines)
    X = convolutional_block(X,  f_hconv=2, f_lconv=5,f_h=1, f_l=3, filters=[256, 256, 1024], stage = 4, block='a', s_h=1, s_l = 2)
    # Stage 5 (≈3 lines)
    #X = convolutional_block(X, f_hconv=1, f_lconv=5, f_h=1, f_l = 3, filters = [512, 512, 2048], stage = 5,block='a', s_h=1, s_l = 2)
    # AVGPOOL (≈1 line). Use "X = AveragePooling2D(...)(X)"
    X = AveragePooling2D(pool_size=(1, 3), strides=None, padding='valid',name='avg_pool')(X)
        # output layer
    X = Flatten()(X)
    if (classes==2):
        X = Dense(1, activation='sigmoid', name='fc-sigmoid')(X)
    else:
        X = Dense(classes, activation='softmax', name='fc-softmax' + str(classes),kernel_initializer = glorot_uniform(seed=0))(X)
    # Create model           
    model = Model(inputs = X_input, outputs = X, name='solo_conv')

    return model


def light_model_16_1000(input_shape , reg, classes = 2):
    
    l2_lambda=reg
    X_input = Input(input_shape)
    
    F1, F2, F3, F4 = [10,10,20,20]
    
    # Save the input value
    X = X_input
    # First component of main path
    X = Conv2D(F1, (4, 10), strides = (1,2), padding = 'valid', 
               name = 'conv-a-'+ str(F1) + 'Filters' + str(4)+ 'x' + str(6) + '-strides' + str(1) + 'x' +str(3)+'-valid',
               kernel_initializer = glorot_uniform(seed=0), kernel_regularizer=regularizers.l2(l2_lambda))(X)
    X = BatchNormalization(axis = 3, name = 'bn-a')(X)
    X = Activation('relu')(X)
    
    X = Conv2D(F2, (4, 15), strides = (1,3), padding = 'valid', 
               name = 'conv-b-'+ str(F2) + 'Filters' + str(4)+ 'x' + str(15) + '-strides' + str(1) + 'x' +str(3)+'-valid',
               kernel_initializer = glorot_uniform(seed=0), kernel_regularizer=regularizers.l2(l2_lambda))(X)
    X = BatchNormalization(axis = 3, name = 'bn-b')(X)
    X = Activation('relu')(X)
    
    X = Conv2D(F3, (4, 10), strides = (1,3), padding = 'valid', 
               name = 'conv-c-'+ str(F3) + 'Filters' + str(4)+ 'x' + str(10) + '-strides' + str(1) + 'x' +str(3)+'-valid',
               kernel_initializer = glorot_uniform(seed=0), kernel_regularizer=regularizers.l2(l2_lambda))(X)
    X = BatchNormalization(axis = 3, name ='bn-c')(X)
    X = Activation('relu')(X)
    X = Conv2D(F4, (4, 8), strides = (1,3), padding = 'valid', 
               name = 'conv-d-'+ str(F4) + 'Filters' + str(4)+ 'x' + str(8) + '-strides' + str(1) + 'x' +str(3)+'-valid',
               kernel_initializer = glorot_uniform(seed=0), kernel_regularizer=regularizers.l2(l2_lambda))(X)
    X = BatchNormalization(axis = 3, name = 'bn-d')(X)
    X = Activation('relu')(X)
    X = Flatten()(X)
    if (classes==2):
        X = Dense(1, activation='sigmoid', name='fc-sigmoid',kernel_initializer = glorot_uniform(seed=0))(X)
    else:
        X = Dense(classes, activation='softmax', name='fc-softmax' + str(classes),kernel_initializer = glorot_uniform(seed=0))(X)
    # Create model           
    model = Model(inputs = X_input, outputs = X, name='solo_conv')

    
    return model



def light_model_32_300(input_shape , reg, drop, classes = 2):
    
    l2_lambda=reg
    drop_prob=drop
    
    X_input = Input(input_shape)
    
    F1, F2, F3, F4,F5 = [10,10,20,20,20]
    
    # Save the input value
    X = X_input
    # First component of main path
    X = Conv2D(F1, (8, 10), strides = (1,2), padding = 'valid', 
               name = 'conv-a-'+ str(F1) + 'Filters' + str(8)+ 'x' + str(10) + '-strides' + str(1) + 'x' +str(2)+'-valid',
               kernel_initializer = glorot_uniform(seed=0), kernel_regularizer=regularizers.l2(l2_lambda))(X)
    X = BatchNormalization(axis = 3, name = 'bn-a')(X)
    X = Activation('relu')(X)
    X= Dropout(drop_prob)(X)
    
    X = Conv2D(F2, (6, 10), strides = (1,2), padding = 'valid', 
               name = 'conv-b-'+ str(F2) + 'Filters' + str(6)+ 'x' + str(15) + '-strides' + str(1) + 'x' +str(2)+'-valid',
               kernel_initializer = glorot_uniform(seed=0), kernel_regularizer=regularizers.l2(l2_lambda))(X)
    X = BatchNormalization(axis = 3, name = 'bn-b')(X)
    X = Activation('relu')(X)
    X= Dropout(drop_prob)(X)
    
    X = Conv2D(F3, (6, 10), strides = (1,2), padding = 'valid', 
               name = 'conv-c-'+ str(F3) + 'Filters' + str(6)+ 'x' + str(10) + '-strides' + str(1) + 'x' +str(2)+'-valid',
               kernel_initializer = glorot_uniform(seed=0), kernel_regularizer=regularizers.l2(l2_lambda))(X)
    X = BatchNormalization(axis = 3, name ='bn-c')(X)
    X = Activation('relu')(X)
    X= Dropout(drop_prob)(X)
    
    X = Conv2D(F4, (4, 8), strides = (1,2), padding = 'valid', 
               name = 'conv-d-'+ str(F4) + 'Filters' + str(4)+ 'x' + str(8) + '-strides' + str(1) + 'x' +str(2)+'-valid',
               kernel_initializer = glorot_uniform(seed=0), kernel_regularizer=regularizers.l2(l2_lambda))(X)
    X = BatchNormalization(axis = 3, name = 'bn-d')(X)
    X = Activation('relu')(X)
    X= Dropout(drop_prob)(X)
    
    X = Conv2D(F5, (5, 5), strides = (1,1), padding = 'valid', 
               name = 'conv-e-'+ str(F5) + 'Filters' + str(5)+ 'x' + str(5) + '-strides' + str(1) + 'x' +str(1)+'-valid',
               kernel_initializer = glorot_uniform(seed=0), kernel_regularizer=regularizers.l2(l2_lambda))(X)
    X = BatchNormalization(axis = 3, name = 'bn-e')(X)
    X = Activation('relu')(X)
    X= Dropout(drop_prob)(X)
    
    X = Flatten()(X)
    if (classes==2):
        X = Dense(1, activation='sigmoid', name='fc-sigmoid',kernel_initializer = glorot_uniform(seed=0))(X)
    else:
        X = Dense(classes, activation='softmax', name='fc-softmax' + str(classes),kernel_initializer = glorot_uniform(seed=0))(X)
    # Create model           
    model = Model(inputs = X_input, outputs = X, name='solo_conv')

    
    return model


def light_model_64_400(input_shape , reg, drop, classes = 2):
    
    drop_prob=drop
    l2_lambda=reg
    X_input = Input(input_shape)
    
    F1, F2, F3, F4,F5 = [10,10,20,20,20]
    
    # Save the input value
    X = X_input
    # First component of main path
    X = Conv2D(F1, (8, 10), strides = (2,3), padding = 'valid', 
               name = 'conv-a-'+ str(F1) + 'Filters' + str(8)+ 'x' + str(10) + '-strides' + str(2) + 'x' +str(3)+'-valid',
               kernel_initializer = glorot_uniform(seed=0), kernel_regularizer=regularizers.l2(l2_lambda))(X)
    X = BatchNormalization(axis = 3, name = 'bn-a')(X)
    X = Activation('relu')(X)
    X= Dropout(drop_prob)(X)
    
    X = Conv2D(F2, (8, 10), strides = (1,2), padding = 'valid', 
               name = 'conv-b-'+ str(F2) + 'Filters' + str(8)+ 'x' + str(15) + '-strides' + str(1) + 'x' +str(2)+'-valid',
               kernel_initializer = glorot_uniform(seed=0), kernel_regularizer=regularizers.l2(l2_lambda))(X)
    X = BatchNormalization(axis = 3, name = 'bn-b')(X)
    X = Activation('relu')(X)
    X= Dropout(drop_prob)(X)
    
    X = Conv2D(F3, (8, 10), strides = (1,2), padding = 'valid', 
               name = 'conv-c-'+ str(F3) + 'Filters' + str(8)+ 'x' + str(10) + '-strides' + str(1) + 'x' +str(2)+'-valid',
               kernel_initializer = glorot_uniform(seed=0), kernel_regularizer=regularizers.l2(l2_lambda))(X)
    X = BatchNormalization(axis = 3, name ='bn-c')(X)
    X = Activation('relu')(X)
    X= Dropout(drop_prob)(X)
    
    X = Conv2D(F4, (6, 8), strides = (1,2), padding = 'valid', 
               name = 'conv-d-'+ str(F4) + 'Filters' + str(6)+ 'x' + str(8) + '-strides' + str(1) + 'x' +str(2)+'-valid',
               kernel_initializer = glorot_uniform(seed=0), kernel_regularizer=regularizers.l2(l2_lambda))(X)
    X = BatchNormalization(axis = 3, name = 'bn-d')(X)
    X = Activation('relu')(X)
    X= Dropout(drop_prob)(X)
    
    X = Conv2D(F5, (5, 5), strides = (1,1), padding = 'valid', 
               name = 'conv-e-'+ str(F5) + 'Filters' + str(5)+ 'x' + str(5) + '-strides' + str(1) + 'x' +str(1)+'-valid',
               kernel_initializer = glorot_uniform(seed=0), kernel_regularizer=regularizers.l2(l2_lambda))(X)
    X = BatchNormalization(axis = 3, name = 'bn-e')(X)
    X = Activation('relu')(X)
    X= Dropout(drop_prob)(X)
    
    X = Flatten()(X)
    if (classes==2):
        X = Dense(1, activation='sigmoid', name='fc-sigmoid',kernel_initializer = glorot_uniform(seed=0))(X)
    else:
        X = Dense(classes, activation='softmax', name='fc-softmax' + str(classes),kernel_initializer = glorot_uniform(seed=0))(X)
    # Create model           
    model = Model(inputs = X_input, outputs = X, name='solo_conv')

    
    return model



def light_model_pp_32_300(input_shape , reg, drop, classes = 2):
    
    l2_lambda=reg
    drop_prob=drop
    
    X_input = Input(input_shape)
    
    F1, F2, F3, F4,F5,F6,F7 = [50,50,80,100,80,50,20]
    
    # Save the input value
    X = X_input
    # First component of main path
    X = Conv2D(F1, (8, 10), strides = (1,2), padding = 'valid', 
               name = 'conv-a-'+ str(F1) + 'Filters' + str(8)+ 'x' + str(10) + '-strides' + str(1) + 'x' +str(2)+'-valid',
               kernel_initializer = glorot_uniform(seed=0), kernel_regularizer=regularizers.l2(l2_lambda))(X)
    X = BatchNormalization(axis = 3, name = 'bn-a')(X)
    X = Activation('relu')(X)
    X= Dropout(drop_prob)(X)
    
    X = Conv2D(F2, (6, 10), strides = (1,2), padding = 'valid', 
               name = 'conv-b-'+ str(F2) + 'Filters' + str(6)+ 'x' + str(15) + '-strides' + str(1) + 'x' +str(2)+'-valid',
               kernel_initializer = glorot_uniform(seed=0), kernel_regularizer=regularizers.l2(l2_lambda))(X)
    X = BatchNormalization(axis = 3, name = 'bn-b')(X)
    X = Activation('relu')(X)
    X= Dropout(drop_prob)(X)
    
    X = Conv2D(F3, (6, 10), strides = (1,2), padding = 'valid', 
               name = 'conv-c-'+ str(F3) + 'Filters' + str(6)+ 'x' + str(10) + '-strides' + str(1) + 'x' +str(2)+'-valid',
               kernel_initializer = glorot_uniform(seed=0), kernel_regularizer=regularizers.l2(l2_lambda))(X)
    X = BatchNormalization(axis = 3, name ='bn-c')(X)
    X = Activation('relu')(X)
    X= Dropout(drop_prob)(X)
    
    X = Conv2D(F4, (4, 8), strides = (1,2), padding = 'valid', 
               name = 'conv-d-'+ str(F4) + 'Filters' + str(4)+ 'x' + str(8) + '-strides' + str(1) + 'x' +str(2)+'-valid',
               kernel_initializer = glorot_uniform(seed=0), kernel_regularizer=regularizers.l2(l2_lambda))(X)
    X = BatchNormalization(axis = 3, name = 'bn-d')(X)
    X = Activation('relu')(X)
    X= Dropout(drop_prob)(X)
    
    X = Conv2D(F5, (5, 5), strides = (1,1), padding = 'valid', 
               name = 'conv-e-'+ str(F5) + 'Filters' + str(5)+ 'x' + str(5) + '-strides' + str(1) + 'x' +str(1)+'-valid',
               kernel_initializer = glorot_uniform(seed=0), kernel_regularizer=regularizers.l2(l2_lambda))(X)
    X = BatchNormalization(axis = 3, name = 'bn-e')(X)
    X = Activation('relu')(X)
    X= Dropout(drop_prob)(X)
    
    X = Conv2D(F6, (3, 3), strides = (1,1), padding = 'valid', 
               name = 'conv-f-'+ str(F6) + 'Filters' + str(3)+ 'x' + str(3) + '-strides' + str(1) + 'x' +str(1)+'-valid',
               kernel_initializer = glorot_uniform(seed=0), kernel_regularizer=regularizers.l2(l2_lambda))(X)
    X = BatchNormalization(axis = 3, name = 'bn-f')(X)
    X = Activation('relu')(X)
    X= Dropout(drop_prob)(X)
    
    X = Conv2D(F7, (2, 2), strides = (1,1), padding = 'valid', 
               name = 'conv-g-'+ str(F7) + 'Filters' + str(2)+ 'x' + str(2) + '-strides' + str(1) + 'x' +str(1)+'-valid',
               kernel_initializer = glorot_uniform(seed=0), kernel_regularizer=regularizers.l2(l2_lambda))(X)
    X = BatchNormalization(axis = 3, name = 'bn-g')(X)
    X = Activation('relu')(X)
    X= Dropout(drop_prob)(X)
    
    X = Flatten()(X)
    if (classes==2):
        X = Dense(1, activation='sigmoid', name='fc-sigmoid',kernel_initializer = glorot_uniform(seed=0))(X)
    else:
        X = Dense(classes, activation='softmax', name='fc-softmax' + str(classes),kernel_initializer = glorot_uniform(seed=0))(X)
    # Create model           
    model = Model(inputs = X_input, outputs = X, name='solo_conv')

    
    return model

def light_model_pp_48_400(input_shape , reg, drop, classes = 2):
    
    drop_prob=drop
    l2_lambda=reg
    X_input = Input(input_shape)
    
    F1, F2, F3, F4,F5,F6,F7 = [50,50,80,100,80,50,20]

    
    # Save the input value
    X = X_input
    # First component of main path
    X = Conv2D(F1, (8, 10), strides = (2,3), padding = 'valid', 
               name = 'conv-a-'+ str(F1) + 'Filters' + str(8)+ 'x' + str(10) + '-strides' + str(2) + 'x' +str(3)+'-valid',
               kernel_initializer = glorot_uniform(seed=0), kernel_regularizer=regularizers.l2(l2_lambda))(X)
    X = BatchNormalization(axis = 3, name = 'bn-a')(X)
    X = Activation('relu')(X)
    X= Dropout(drop_prob)(X)
    
    X = Conv2D(F2, (8, 10), strides = (1,2), padding = 'valid', 
               name = 'conv-b-'+ str(F2) + 'Filters' + str(8)+ 'x' + str(15) + '-strides' + str(1) + 'x' +str(2)+'-valid',
               kernel_initializer = glorot_uniform(seed=0), kernel_regularizer=regularizers.l2(l2_lambda))(X)
    X = BatchNormalization(axis = 3, name = 'bn-b')(X)
    X = Activation('relu')(X)
    X= Dropout(drop_prob)(X)
    
    X = Conv2D(F3, (6, 10), strides = (1,2), padding = 'valid', 
               name = 'conv-c-'+ str(F3) + 'Filters' + str(6)+ 'x' + str(10) + '-strides' + str(1) + 'x' +str(2)+'-valid',
               kernel_initializer = glorot_uniform(seed=0), kernel_regularizer=regularizers.l2(l2_lambda))(X)
    X = BatchNormalization(axis = 3, name ='bn-c')(X)
    X = Activation('relu')(X)
    X= Dropout(drop_prob)(X)
    
    X = Conv2D(F4, (4, 8), strides = (1,2), padding = 'valid', 
               name = 'conv-d-'+ str(F4) + 'Filters' + str(6)+ 'x' + str(8) + '-strides' + str(1) + 'x' +str(2)+'-valid',
               kernel_initializer = glorot_uniform(seed=0), kernel_regularizer=regularizers.l2(l2_lambda))(X)
    X = BatchNormalization(axis = 3, name = 'bn-d')(X)
    X = Activation('relu')(X)
    X= Dropout(drop_prob)(X)
    
    X = Conv2D(F5, (3, 5), strides = (1,1), padding = 'valid', 
               name = 'conv-e-'+ str(F5) + 'Filters' + str(3)+ 'x' + str(5) + '-strides' + str(1) + 'x' +str(1)+'-valid',
               kernel_initializer = glorot_uniform(seed=0), kernel_regularizer=regularizers.l2(l2_lambda))(X)
    X = BatchNormalization(axis = 3, name = 'bn-e')(X)
    X = Activation('relu')(X)
    X= Dropout(drop_prob)(X)
    
    X = Conv2D(F6, (2, 2), strides = (1,1), padding = 'valid', 
               name = 'conv-f-'+ str(F6) + 'Filters' + str(2)+ 'x' + str(2) + '-strides' + str(1) + 'x' +str(1)+'-valid',
               kernel_initializer = glorot_uniform(seed=0), kernel_regularizer=regularizers.l2(l2_lambda))(X)
    X = BatchNormalization(axis = 3, name = 'bn-f')(X)
    X = Activation('relu')(X)
    X= Dropout(drop_prob)(X)
    
    X = Conv2D(F7, (1, 1), strides = (1,1), padding = 'valid', 
               name = 'conv-g-'+ str(F7) + 'Filters' + str(1)+ 'x' + str(1) + '-strides' + str(1) + 'x' +str(1)+'-valid',
               kernel_initializer = glorot_uniform(seed=0), kernel_regularizer=regularizers.l2(l2_lambda))(X)
    X = BatchNormalization(axis = 3, name = 'bn-g')(X)
    X = Activation('relu')(X)
    X= Dropout(drop_prob)(X)
    
    
    X = Flatten()(X)
    if (classes==2):
        X = Dense(1, activation='sigmoid', name='fc-sigmoid',kernel_initializer = glorot_uniform(seed=0))(X)
    else:
        X = Dense(classes, activation='softmax', name='fc-softmax' + str(classes),kernel_initializer = glorot_uniform(seed=0))(X)
    # Create model           
    model = Model(inputs = X_input, outputs = X, name='solo_conv')

    
    return model



def light_model_pp_64_400(input_shape , reg, drop, classes = 2):
    
    drop_prob=drop
    l2_lambda=reg
    X_input = Input(input_shape)
    
    F1, F2, F3, F4,F5,F6,F7 = [50,50,80,100,80,50,20]

    
    # Save the input value
    X = X_input
    # First component of main path
    X = Conv2D(F1, (8, 10), strides = (2,3), padding = 'valid', 
               name = 'conv-a-'+ str(F1) + 'Filters' + str(8)+ 'x' + str(10) + '-strides' + str(2) + 'x' +str(3)+'-valid',
               kernel_initializer = glorot_uniform(seed=0), kernel_regularizer=regularizers.l2(l2_lambda))(X)
    X = BatchNormalization(axis = 3, name = 'bn-a')(X)
    X = Activation('relu')(X)
    X= Dropout(drop_prob)(X)
    
    X = Conv2D(F2, (8, 10), strides = (1,2), padding = 'valid', 
               name = 'conv-b-'+ str(F2) + 'Filters' + str(8)+ 'x' + str(15) + '-strides' + str(1) + 'x' +str(2)+'-valid',
               kernel_initializer = glorot_uniform(seed=0), kernel_regularizer=regularizers.l2(l2_lambda))(X)
    X = BatchNormalization(axis = 3, name = 'bn-b')(X)
    X = Activation('relu')(X)
    X= Dropout(drop_prob)(X)
    
    X = Conv2D(F3, (8, 10), strides = (1,2), padding = 'valid', 
               name = 'conv-c-'+ str(F3) + 'Filters' + str(8)+ 'x' + str(10) + '-strides' + str(1) + 'x' +str(2)+'-valid',
               kernel_initializer = glorot_uniform(seed=0), kernel_regularizer=regularizers.l2(l2_lambda))(X)
    X = BatchNormalization(axis = 3, name ='bn-c')(X)
    X = Activation('relu')(X)
    X= Dropout(drop_prob)(X)
    
    X = Conv2D(F4, (6, 8), strides = (1,2), padding = 'valid', 
               name = 'conv-d-'+ str(F4) + 'Filters' + str(6)+ 'x' + str(8) + '-strides' + str(1) + 'x' +str(2)+'-valid',
               kernel_initializer = glorot_uniform(seed=0), kernel_regularizer=regularizers.l2(l2_lambda))(X)
    X = BatchNormalization(axis = 3, name = 'bn-d')(X)
    X = Activation('relu')(X)
    X= Dropout(drop_prob)(X)
    
    X = Conv2D(F5, (5, 5), strides = (1,1), padding = 'valid', 
               name = 'conv-e-'+ str(F5) + 'Filters' + str(5)+ 'x' + str(5) + '-strides' + str(1) + 'x' +str(1)+'-valid',
               kernel_initializer = glorot_uniform(seed=0), kernel_regularizer=regularizers.l2(l2_lambda))(X)
    X = BatchNormalization(axis = 3, name = 'bn-e')(X)
    X = Activation('relu')(X)
    X= Dropout(drop_prob)(X)
    
    X = Conv2D(F6, (2, 2), strides = (1,1), padding = 'valid', 
               name = 'conv-f-'+ str(F6) + 'Filters' + str(2)+ 'x' + str(2) + '-strides' + str(1) + 'x' +str(1)+'-valid',
               kernel_initializer = glorot_uniform(seed=0), kernel_regularizer=regularizers.l2(l2_lambda))(X)
    X = BatchNormalization(axis = 3, name = 'bn-f')(X)
    X = Activation('relu')(X)
    X= Dropout(drop_prob)(X)
    
    X = Conv2D(F7, (1, 1), strides = (1,1), padding = 'valid', 
               name = 'conv-g-'+ str(F7) + 'Filters' + str(1)+ 'x' + str(1) + '-strides' + str(1) + 'x' +str(1)+'-valid',
               kernel_initializer = glorot_uniform(seed=0), kernel_regularizer=regularizers.l2(l2_lambda))(X)
    X = BatchNormalization(axis = 3, name = 'bn-g')(X)
    X = Activation('relu')(X)
    X= Dropout(drop_prob)(X)
    
    
    X = Flatten()(X)
    if (classes==2):
        X = Dense(1, activation='sigmoid', name='fc-sigmoid',kernel_initializer = glorot_uniform(seed=0))(X)
    else:
        X = Dense(classes, activation='softmax', name='fc-softmax' + str(classes),kernel_initializer = glorot_uniform(seed=0))(X)
    # Create model           
    model = Model(inputs = X_input, outputs = X, name='solo_conv')

    
    return model
    

#=>.] - ETA: 17s - loss: 4.9210 - acc: 0.6945
# epoch 1 92250/92250 [==============================] - 32299s 350ms/step - loss: 4.9212 - acc: 0.6945
# epoch 2 92250/92250 [==============================] - 32161s 349ms/step - loss: 4.9224 - acc: 0.6946
#sul test set Mian (quasi sano)
#Loss = 0.23514665683265745
#Test Accuracy = 0.9854110207768745
#Exception ignored in: <bound method BaseSession.__del__ of <tensorflow.python.client.session.Session object at 0x7f8af73e6160>>
#Traceback (most recent call last):
#  File "/home/eugenio/tensorflow/lib/python3.5/site-packages/tensorflow/python/client/session.py", line 696, in __del__
#TypeError: 'NoneType' object is not callable
#Total params: 8,038,785
#Trainable params: 8,002,177
#Non-trainable params: 36,608

