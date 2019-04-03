# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 00:21:13 2019

@author: SHREYANK
"""

import scipy.io as sio 
import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.utils import to_categorical
from sklearn.utils import shuffle
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.utils import plot_model
from keras.models import Model,Sequential
from keras.layers import Input
from keras.layers import Dense
from keras.layers.recurrent import LSTM
from keras.layers.merge import concatenate
import tensorflow as tf

matlab_inputs_A = sio.loadmat('amazon_20110928')

A_features = matlab_inputs_A['hist_features']
A_labels = matlab_inputs_A['hist_features_index']
A_features,A_labels=shuffle(A_features,A_labels,random_state=0)
A_labels=A_labels-1

matlab_inputs_B = sio.loadmat('webcam_20110928')

B_features = matlab_inputs_B['hist_features']
B_labels = matlab_inputs_B['hist_features_index']
B_features,B_labels=shuffle(B_features,B_labels,random_state=0)
B_labels=B_labels-1

matlab_inputs_C = sio.loadmat('dslr_20110928')

C_features = matlab_inputs_C['hist_features']
C_labels = matlab_inputs_C['hist_features_index']
C_features,C_labels=shuffle(C_features,C_labels,random_state=0)
C_labels=C_labels-1

thresh=0.2

xA_train=np.float64(A_features[0:int((1-thresh)*len(A_features)),:])
yA_train=to_categorical(A_labels[0:int((1-thresh)*len(A_features)),:])
xA_test=np.float64(A_features[int((1-thresh)*len(A_features)):,:])
yA_test=to_categorical(A_labels[int((1-thresh)*len(A_features)):,:])


xB_train=np.float64(B_features[0:int((1-thresh)*len(B_features)),:])
yB_train=to_categorical(B_labels[0:int((1-thresh)*len(B_features)),:])
xB_test=np.float64(B_features[int((1-thresh)*len(B_features)):,:])
yB_test=to_categorical(B_labels[int((1-thresh)*len(B_features)):,:])

xC_train=np.float64(C_features[0:int((1-thresh)*len(C_features)),:])
yC_train=to_categorical(C_labels[0:int((1-thresh)*len(C_features)),:])
xC_test=np.float64(C_features[int((1-thresh)*len(C_features)):,:])
yC_test=to_categorical(C_labels[int((1-thresh)*len(C_features)):,:])





#modelA
input1=Input(shape=(800,))
output1=Dense(400,activation='relu')(input1)
model1=Model(input1,output1)

#modelB
input2=Input(shape=(800,))
output2=Dense(400,activation='relu')(input2)
model2=Model(input2,output2)

#modelC
input3=Input(shape=(400,))
output3=Dense(31,activation='softmax')(input3)
model3=Model(input3,output3)


#TRAIN modelA-A
tt=model3(model1(input1))
model = Model(inputs=[input1], outputs=tt)
#plot_model(model, to_file='model.png')
model.compile( optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
early_stopping=EarlyStopping(monitor='val_acc', patience=10, verbose=0, mode='max')
filepath="domain_a_c.h5"
check_point=ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False,mode='max')
# Fit the model
model.fit(xA_train,yA_train,validation_data=[xA_test,yA_test],epochs=100,callbacks=[check_point,early_stopping], batch_size=32)


#TRAIN modelB-B
tt2=model3(model2(input2))
modelbb = Model(inputs=[input2], outputs=tt2)
#plot_model(model, to_file='model.png')
modelbb.compile( optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

early_stopping=EarlyStopping(monitor='val_acc', patience=10, verbose=0, mode='max')
filepath="domain_b_c.h5"
check_point=ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False,mode='max')
# Fit the model
modelbb.fit(xB_train,yB_train,validation_data=[xB_test,yB_test],epochs=100,callbacks=[check_point,early_stopping], batch_size=32)



xA_train_new_domain=model1.predict(xA_train)
xB_train_new_domain=model2.predict(xB_train)

x_train=np.concatenate((xA_train_new_domain,xB_train_new_domain),axis=0)
y_train=np.concatenate((yA_train,yB_train),axis=0)
x_train,y_train=shuffle(x_train,y_train,random_state=0)


#TRAIN modelC-C
tt3=model3(input3)
modelcc = Model(inputs=[input3], outputs=tt3)
#plot_model(model, to_file='model.png')
modelcc.compile( optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

early_stopping=EarlyStopping(monitor='acc', patience=10, verbose=0, mode='max')
filepath="domain_c.h5"
check_point=ModelCheckpoint(filepath, monitor='acc', verbose=1, save_best_only=True, save_weights_only=False,mode='max')
# Fit the model
modelcc.fit(x_train,y_train,epochs=100,callbacks=[check_point,early_stopping], batch_size=32)





xB_test
xA_test
pred_xBA=model3.predict(model1.predict(xB_test))
pred_xAB=model3.predict(model2.predict(xA_test))

pred_xBA=np.argmax(pred_xBA,axis=1)
actual_xBA=np.argmax(yB_test,axis=1)
pred_xAB=np.argmax(pred_xAB,axis=1)
actual_xAB=np.argmax(yA_test,axis=1)

acc_A=actual_xAB-pred_xAB
acc_A=len(np.where(acc_A==0)[0])/len(acc_A)
acc_B=actual_xBA-pred_xBA
acc_B=len(np.where(acc_B==0)[0])/len(acc_B)










