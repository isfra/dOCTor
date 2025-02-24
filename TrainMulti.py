import collections
import csv
import os
import scipy.misc
import time

from keras import layers
from keras.applications.imagenet_utils import preprocess_input
from keras.callbacks import Callback , ModelCheckpoint, TensorBoard
from keras.initializers import glorot_uniform
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils import plot_model
from keras.utils.data_utils import get_file
from keras.utils.vis_utils import model_to_dot
from matplotlib.pyplot import imshow

import DataProcessing as DP
import PlaccaModel as PM
import keras.backend as K
import matplotlib.pyplot as plt
import numpy as np


#model_path="/home/eugenio/workspace/documets di pasqualino/Documents/pasqualino/placcaultras"
model_path="C:\\Users\\Franz\\Desktop\\WiP\\oct_software\\models\\best_model"#"/home/fisici/Documents/OperazionePlacca"

#env_path="/home/eugenio/workspace/documets di pasqualino/Documents/OCTdatabase/all_pullbacks"#/old selsao"
env_path="C:\\Users\\Franz\\Desktop\\WiP\\oct_software\\models\\best_model\\training_set"#"/home/fisici/Documents/OCTdatabase/all_pullbacks"


K.set_image_data_format('channels_last')
K.set_learning_phase(1)

ordereddict =DP. _OrderedDictMaker()

Info = ordereddict[
        'version'        : "4.7-4"    ,
        'database'   : "sel_long",
        'tr_set'         : (0,95)  ,
        'cv_set'         : (95,99) ,
        'te_set'         : (99,100),
        'depth'          : 400     ,
        'degree'         : 64      , 
        'stride'         : 16      ,
        'epochs'         : 120      ,
        'batch_size'     : 32      ,
        'reg'            : 0. ,
        'threshold'      : 0.5     ,
        'nonlipid_weight': 1.      ,
        'lipid_weight'   : 1.      ,
        'train_loss'     : 0       ,
        'train_accuracy' : 0       ,
        'CV_accuracy'    : 0       ,
        'test_accuracy'  : 0       ,
        'TP'             : 0       ,
        'FP'             : 0       ,
        'TN'             : 0       ,
        'FN'             : 0       ,
        'precision'      : 0       ,
        'recall'         : 0       ,
        'f1'             : 0       ,
        'tr_time'        : 0       ,
        'test_time'      : 0       ,
        'note'           :"no guide calcium dropout omogeneo 0.2 new data"
        ]

 
if Info['database'] is "mixed_short":
    database ="/mixedshortnuovi.npy"#"/mixedshortpullbacks.npy"
elif Info['database'] is "mixed_long":
    database ="/mixedpullbacks.npy"
elif Info['database'] is "sel_mixed_long":
    database ="/selmixedpullbacks.npy"
elif Info['database'] is "sel_long":
    database ="/selpullbacks.npy"
elif Info['database'] is "sel_cut_mixed_long":
    database ="/selcutmixedpullbacks.npy"
elif Info['database'] is "all_pullbacks":
    database ="/allpullbacks.npy"


''' qui è il main; quello che fai girare
    si caricano i dati e poi si chiama il modello
    si usano poi le funzioni che possiede per compilare ottimizzare e fare le predizioni
    shape dell input N_h, N_l N_c=1  '''

train_patient = "Aladin_2016_02_19_11_37_21"
test_patient = "Mian_2014_01_02_04_09_54"

#class_weight = {0 : Info['nonlipid_weight'],
#    1: Info['lipid_weight']}

X_train,Y_train,_, _, _  = DP.pullback_n_labels_part_min_multiclass (env_path,database, Info['tr_set'][0],Info['tr_set'][1], depth=Info["depth"],deg=Info['degree'], stride=Info['stride'] )

X_val,Y_val,labels_val,nSlices, nRadius  = DP.pullback_n_labels_part_min_multiclass (env_path,database, Info['cv_set'][0],Info['cv_set'][1], depth=Info["depth"], deg=Info['degree'], stride=Info['stride'] )


'''  dopo ave caricato i dati dobbiamo aggiungere la dim del canale 
     pure se è una sola perche sara la dim dei canali ottenuti da diversi filtri   '''

X_train = np.reshape(X_train,np.append(X_train.shape,1))
X_val = np.reshape(X_val,np.append(X_val.shape,1))


print("number of training examples = " + str(X_train.shape[0]))
print("number of test examples = " + str(X_val.shape[0]))
print("X_train shape: " + str(X_train.shape))
print("Y_train shape: " + str(Y_train.shape))
print("X_val shape: " + str(X_val.shape))
print("Y_val shape: " + str(Y_val.shape))

t=time.time()

class History(Callback):
    def on_train_begin(self, logs={}):
        self.lossesEpoch = []
        self.lossesBatch = []
        self.accEpoch = []
        self.accBatch = []
        self.accEpochCV = []
        self.lossesEpochCV = []
    def on_batch_end(self, batch, logs={}):
        self.accBatch.append(logs.get('acc'))
        self.lossesBatch.append(logs.get('loss'))
    def on_epoch_end(self,epoch,logs={}):
        self.accEpoch.append(logs.get('acc'))
        self.lossesEpoch.append(logs.get('loss'))    
        self.accEpochCV.append(logs.get('val_acc'))
        self.lossesEpochCV.append(logs.get('val_loss'))
        
history =History() 
checkpoint=ModelCheckpoint(model_path + '/PlaccaModel{epoch:02d}.hdf5')
tbCallBack = TensorBoard(log_dir='./Graph1', histogram_freq=1, write_graph=True)#, write_images=True)

#old_model=load_model(model_path+"/OldPlaccaModel.h5")
#model_weights=old_model.get_weights()

model = PM.light_model_pp_64_400(input_shape = X_train.shape[1:4], reg=Info['reg'], drop=0.2, classes = 3)#ResNet50(input_shape = X_train.shape[1:4], classes = 2,reg=Info['reg'])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#model.set_weights(model_weights)

#,class_weight = class_weight
model.fit(x = X_train, y = Y_train, epochs = Info['epochs'], batch_size = Info['batch_size'],callbacks=[history,checkpoint,tbCallBack],validation_data=(X_val,Y_val),shuffle=True)

model.save(model_path + "/PlaccaModel.h5")

Info['tr_time'] = (time.time()-t)/3600

preds_tr= model.predict(X_train,verbose=1)
np.save(model_path + "/Train_prediction.npy", preds_tr)
ev_tr = model.evaluate(X_train, Y_train)
np.save(model_path + "/Train_performance.npy", ev_tr)

preds_val= model.predict(X_val,verbose=1)
np.save(model_path + "/Validation_prediction.npy", preds_val)

del X_train
del Y_train
X_test,Y_test,labels_test,_, _ = DP.pullback_n_labels_part_min_multiclass (env_path,database, Info['te_set'][0],Info['te_set'][1], depth=Info["depth"], deg=Info['degree'], stride=Info['stride'] )
X_test = np.reshape(X_test,np.append(X_test.shape,1))

t=time.time()

preds_te= model.predict(X_test)
np.save(model_path + "/Test_prediction.npy", preds_te)
ev_te = model.evaluate(X_test, Y_test)
np.save(model_path + "/Test_performance.npy", ev_te)

Info['test_time'] = (time.time()-t)/3600

Info['train_loss']=ev_tr[0]
Info['train_accuracy']=ev_tr[1]
Info['CV_accuracy']= history.accEpochCV[-1]
Info['test_accuracy']=ev_te[1]

print ("Loss = " + str(ev_tr[0]))
print ("Train Accuracy = " + str(ev_tr[1]))

print ("Loss = " + str(ev_te[0]))
print ("Test Accuracy = " + str(ev_te[1]))

#preds_te=np.load(model_path+'/Test_prediction.npy')

DP.save_list(history.accEpoch,model_path+"/accEpoch.txt")
DP.save_list(history.accEpochCV,model_path+"/accEpochCV.txt")
DP.save_list(history.lossesEpoch,model_path+"/lossesEpoch.txt")
DP.save_list(history.lossesEpochCV,model_path+"/lossesEpochCV.txt")
DP.save_list(history.accBatch,model_path+"/accBatch.txt")
DP.save_list(history.lossesBatch,model_path+"/lossesBatch.txt")

plt.plot(history.accEpoch , lw=1)
plt.plot(history.accEpochCV, lw=1)
plt.title('epoch accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
#plt.show()
plt.savefig(model_path + '/acc_e.png', bbox_inches='tight')
plt.close()

plt.plot(history.lossesEpoch , lw=1)
plt.plot(history.lossesEpochCV, lw=1)
plt.title('epoch loss')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig(model_path + '/loss_e.png', bbox_inches='tight')
plt.close()
'''
plt.plot(history.accBatch , lw=0.1)
plt.title('batch accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.savefig(model_path + '/acc_b.png', bbox_inches='tight')
plt.close()



plt.plot(history.lossesBatch, lw=0.1)
plt.title('batch loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.savefig(model_path + '/loss_b.png', bbox_inches='tight')
plt.close()
'''
plot_model(model, to_file='model.png',show_shapes=True)

Info,score_rec=DP.evaluate_reconstruction(env_path, model_path, Info, preds_val,labels_val,nSlices, nRadius,3,20)
scores=DP.roc_multiclass(model_path, preds_val, Y_val, 3, 100)
DP.best_f1s(scores)
DP.best_dor(scores)

DP.WriteOrdDictToCSV(model_path + '/info.csv',Info)       

print(Info)
