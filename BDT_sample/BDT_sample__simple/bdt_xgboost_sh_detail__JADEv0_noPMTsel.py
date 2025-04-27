import xgboost as xgb
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import torch
import h5py
from tqdm import tqdm
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft
import sys, os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--depth', action='store', type=int, default=45, help='max_depth')
parser.add_argument('--estimators', action='store', type=int, default=100, help='n_estimators')
parser.add_argument('--lr', action='store', type=float, default=0.1, help='learning rate')
parser.add_argument('--random_states', action='store', type=int, default=42, help='random_state')
parser.add_argument('--test_size', action='store', type=float, default=0, help='test_size')
args = parser.parse_args()



########################################################################

cw_dir_all="/store/hep/users/wonsang1995/comdata_pCharge_cut" ## jade v0

wf_all_1592 = []
label_all_1592 = []
#Npmt_1592_ME = []
#Npmt_1592_FN = []
subTrig = []

vtxX_all_1592 = []
vtxY_all_1592 = []
vtxZ_all_1592 = []

########################

pulse_check = -1 #0~3


for i in tqdm(range(6157)):
#for i in tqdm(range(20)):
    fName_ME=cw_dir_all+"/r001592/ME/debug.r001592.f"+"{0:05d}".format(i)+".h5"

    fin_ME = h5py.File(fName_ME, 'r', libver='latest', swmr=True)
    #if i==0 : print(fin_ME["events"].keys())
    tree_ME = fin_ME['events']
    wf_ME = np.array(tree_ME["waveform"])
    wfmask_ME = (wf_ME > -1e9)
    valid_ME = np.array(tree_ME["validPMT"])
    charge_ME = np.array(tree_ME["pmtQ"])
    #Npmt_ME = np.array(tree_ME["Npmt"])
    trigID_ME = np.array(tree_ME["trigID"])
    subrun_ME = np.array(tree_ME["subrun"])

    vtxX_ME = np.array( tree_ME["vertexX"] )
    vtxY_ME = np.array( tree_ME["vertexY"] )
    vtxZ_ME = np.array( tree_ME["vertexZ"] )
    
    pmtQ_ME = charge_ME*valid_ME
    #NPMT_ME = Npmt_ME*(np.sum(valid_ME,axis=1)>0)
    subTrig_ME = (subrun_ME*1e6+trigID_ME)*(np.sum(valid_ME,axis=1)>0)

    #print("pmtQ_ME : ")
    #print(pmtQ_ME.shape)
    
    sumQ_ME = pmtQ_ME.sum(axis=1)
    fracQ_ME = pmtQ_ME/sumQ_ME[:,np.newaxis]
    fracQ_ME = np.nan_to_num( fracQ_ME, nan=0 )
    wf_ME /= wf_ME.max(axis=2)[:,:,np.newaxis]
    np.nan_to_num(wf_ME, copy=False, nan=0.0)
    wf_ME *= wfmask_ME
    #wf_ME = wf_ME*fracQ_ME[:,:,np.newaxis]
    pulse_ME = wf_ME.sum(axis=1)/96
    label_ME = np.ones(len(pulse_ME))

    pulse_ME = pulse_ME[:,40:]

    wf_all_1592 += pulse_ME.tolist()
    label_all_1592 += label_ME.tolist()
    #Npmt_1592_ME += NPMT_ME.tolist()
    subTrig += subTrig_ME.tolist()

    vtxX_all_1592 += vtxX_ME.tolist()
    vtxY_all_1592 += vtxY_ME.tolist()
    vtxZ_all_1592 += vtxZ_ME.tolist()
    
    wf_ME, valid_ME, charge_ME, pmtQ_ME, sumQ_ME, fracQ_ME, pulse_ME, label_ME = [],[],[],[],[],[],[],[]
    vtxX_ME, vtxY_ME, vtxZ_ME = [], [], []


#plt.hist(Npmt_1592_ME,bins=96,range=[0,97],histtype="step")
#plt.xlabel("N_pmt")
#plt.show()

############################
for i in tqdm(range(6157)):
#for i in tqdm(range(20)):
    fName_FN=cw_dir_all+"/r001592/FN/debug.r001592.f"+"{0:05d}".format(i)+".h5"

    fin_FN = h5py.File(fName_FN, 'r', libver='latest', swmr=True)
    #if i==0 : print(fin_FN.keys())
    tree_FN = fin_FN['events']
    wf_FN = np.array(tree_FN["waveform"])
    wfmask_FN = (wf_FN > -1e9)
    valid_FN = np.array(tree_FN["validPMT"])
    charge_FN = np.array(tree_FN["pmtQ"])
    #Npmt_FN = np.array(tree_FN["Npmt"])
    trigID_FN = np.array(tree_FN["trigID"])
    subrun_FN = np.array(tree_FN["subrun"])

    vtxX_FN = np.array( tree_FN["vertexX"] )
    vtxY_FN = np.array( tree_FN["vertexY"] )
    vtxZ_FN = np.array( tree_FN["vertexZ"] )

    pmtQ_FN = charge_FN*valid_FN
    #NPMT_FN = Npmt_FN*(np.sum(valid_FN,axis=1)>0)
    subTrig_FN = (subrun_FN*1e6+trigID_FN)*(np.sum(valid_FN,axis=1)>0)

    sumQ_FN = pmtQ_FN.sum(axis=1)
    fracQ_FN = pmtQ_FN/sumQ_FN[:,np.newaxis]
    fracQ_FN = np.nan_to_num( fracQ_FN, nan=0 )
    wf_FN /= wf_FN.max(axis=2)[:,:,np.newaxis]
    np.nan_to_num(wf_FN, copy=False, nan=0.0)
    wf_FN *= wfmask_FN
    #wf_FN = wf_FN*fracQ_FN[:,:,np.newaxis]
    pulse_FN = wf_FN.sum(axis=1)/96
    label_FN = np.zeros(len(pulse_FN))
    #print("pusle_FN :")
    #print(pulse_FN.shape)

    pulse_FN = pulse_FN[:,40:]

    wf_all_1592 += pulse_FN.tolist()
    label_all_1592 += label_FN.tolist()
    #Npmt_1592_FN += NPMT_FN.tolist()
    subTrig += subTrig_FN.tolist()

    vtxX_all_1592 += vtxX_FN.tolist()
    vtxY_all_1592 += vtxY_FN.tolist()
    vtxZ_all_1592 += vtxZ_FN.tolist()

    wf_FN, valid_FN, charge_FN, pmtQ_FN, sumQ_FN, fracQ_FN, pulse_FN, label_FN = [],[],[],[],[],[],[],[]
    vtxX_FN, vtxY_FN, vtxZ_FN = [], [], []


#plt.hist(Npmt_1592_FN,bins=96,range=[0,97],histtype="step")
#plt.xlabel("N_pmt")
#plt.show()


######################
########################################################################
########################################################################

wf_all_1563 = []
label_all_1563 = []

########################

for i in tqdm(range(4869)):
#for i in tqdm(range(20)):
    fName_ME=cw_dir_all+"/r001563/ME/debug.r001563.f"+"{0:05d}".format(i)+".h5"

    fin_ME = h5py.File(fName_ME, 'r', libver='latest', swmr=True)
    tree_ME = fin_ME['events']
    wf_ME = np.array(tree_ME["waveform"])
    wfmask_ME = (wf_ME > -1e9)
    valid_ME = np.array(tree_ME["validPMT"])
    charge_ME = np.array(tree_ME["pmtQ"])
    pmtQ_ME = charge_ME*valid_ME
    sumQ_ME = pmtQ_ME.sum(axis=1)
    fracQ_ME = pmtQ_ME/sumQ_ME[:,np.newaxis]
    fracQ_ME = np.nan_to_num( fracQ_ME, nan=0 )
    wf_ME /= wf_ME.max(axis=2)[:,:,np.newaxis]
    np.nan_to_num(wf_ME, copy=False, nan=0.0)
    wf_ME *= wfmask_ME
    #wf_ME = wf_ME*fracQ_ME[:,:,np.newaxis]
    pulse_ME = wf_ME.sum(axis=1)/96
    label_ME = np.ones(len(pulse_ME))
    #print("pusle_ME :")
    #print(pulse_ME.shape)

    pulse_ME = pulse_ME[:,40:]
    wf_all_1563 += pulse_ME.tolist()
    label_all_1563 += label_ME.tolist()

    wf_ME, valid_ME, charge_ME, pmtQ_ME, sumQ_ME, fracQ_ME, pulse_ME, label_ME = [],[],[],[],[],[],[],[]



############################
for i in tqdm(range(4869)):
#for i in tqdm(range(20)):
    fName_FN=cw_dir_all+"/r001563/FN/debug.r001563.f"+"{0:05d}".format(i)+".h5"

    fin_FN = h5py.File(fName_FN, 'r', libver='latest', swmr=True)
    tree_FN = fin_FN['events']
    wf_FN = np.array(tree_FN["waveform"])
    wfmask_FN = (wf_FN > -1e9)
    valid_FN = np.array(tree_FN["validPMT"])
    charge_FN = np.array(tree_FN["pmtQ"])
    pmtQ_FN = charge_FN*valid_FN
    sumQ_FN = pmtQ_FN.sum(axis=1)
    fracQ_FN = pmtQ_FN/sumQ_FN[:,np.newaxis]
    fracQ_FN = np.nan_to_num( fracQ_FN, nan=0 )
    wf_FN /= wf_FN.max(axis=2)[:,:,np.newaxis]
    np.nan_to_num(wf_FN, copy=False, nan=0.0)
    wf_FN *= wfmask_FN
    #wf_FN = wf_FN*fracQ_FN[:,:,np.newaxis]
    pulse_FN = wf_FN.sum(axis=1)/96
    label_FN = np.zeros(len(pulse_FN))
    #print("pusle_FN :")
    #print(pulse_FN.shape)

    wf_all_1563 += pulse_FN.tolist()
    label_all_1563 += label_FN.tolist()

    wf_FN, valid_FN, charge_FN, pmtQ_FN, sumQ_FN, fracQ_FN, pulse_FN, label_FN = [],[],[],[],[],[],[],[]

######################
########################################################################

print("run 1592 : ")
print(np.array(wf_all_1592).shape)
print(np.array(label_all_1592).shape)

print("run 1563 : ")
print(np.array(wf_all_1563).shape)
print(np.array(label_all_1563).shape)

import numpy as np

#print(y)
# 데이터 분할 (학습 및 테스트 데이터)
X_train, X_test, y_train, y_test = train_test_split(wf_all_1563, label_all_1563, random_state=args.random_states,test_size=args.test_size)
63

print("X_train")
print(np.array(X_train).shape)

import math
import time
start = time.time()

label_all_1563 = np.array(label_all_1563)
pos_weight = np.sum(label_all_1563==1)/np.sum(label_all_1563==0)
evals = [(X_train, y_train), (X_test, y_test)]

print("# of ME (run1563) = ",np.sum(label_all_1563==1))
print("# of FN (run1563) = ",np.sum(label_all_1563==0))
print("pos_weight = ",pos_weight)


# Boosted Decision Tree 모델 생성
boosted_model = xgb.XGBClassifier(max_depth=args.depth, n_estimators=args.estimators, learning_rate=args.lr, random_state=args.random_states, subsample=0.5, eval_metric='logloss', scale_pos_weight = pos_weight)#, early_stopping_rounds=50)
model_str="""
boosted_model = xgb.XGBClassifier(max_depth=%d, n_estimators=%d, learning_rate=%.1f, random_state=%d, test_size=%.3f, scale_pos_weight=%.2f)
"""%(args.depth, args.estimators, args.lr, args.random_states, args.test_size, pos_weight)
print(model_str)

# 모델 학습
print("\ntrain_start")
boosted_model.fit(X_train, y_train, eval_set=evals, verbose=True )#, eval_metric="log_loss")
print("train_end\n")


end = time.time()
print( "training time : " )
print(f"{end - start:.5f} sec")


# 예측
y_pred_1592 = boosted_model.predict(wf_all_1592)

start_pred = time.time()
y_pred_proba_1592 = boosted_model.predict_proba(wf_all_1592)
end_pred = time.time()
print( "evaluation time : " )
print(f"{end_pred - start_pred:.5f} sec")

accuracy = accuracy_score(label_all_1592, y_pred_1592)
report = classification_report(label_all_1592, y_pred_1592)
print(f"Accuracy: {accuracy}")
print("Classification Report:\n", report)


label_all_1592 = np.array(label_all_1592)
wf_all_1592 = np.array(wf_all_1592)

eventWeight = label_all_1592 * ( np.sum(label_all_1592<1)/len(label_all_1592) ) + np.abs(1-label_all_1592)
#print("eventWeight: ", eventWeight)

import pandas as pd


df = pd.DataFrame({"pred":y_pred_proba_1592[:,1], "label":label_all_1592, "weight":eventWeight, "subTrig":subTrig, "vtxX":vtxX_all_1592, "vtxY":vtxY_all_1592,"vtxZ":vtxZ_all_1592})


df.to_csv("bdt__xgboost__depth_%d__estimator_%d__lr_%.1f__random_%d__test_size_%.3f__noPedestal__JADEv0_noPMTsel.csv"%(args.depth, args.estimators, args.lr, args.random_states,args.test_size))


evals_result  = boosted_model.evals_result()
train_logloss = evals_result["validation_0"]["logloss"]
val_logloss = evals_result["validation_1"]["logloss"]
df_train = pd.DataFrame({"train_loss":train_logloss, "val_loss":val_logloss})
df_train.to_csv("train_bdt__xgboost__depth_%d__estimator_%d__lr_%.1f__random_%d__test_size_%.3f__noPedestal__JADEv0_noPMTsel.csv"%(args.depth, args.estimators, args.lr, args.random_states,args.test_size))


