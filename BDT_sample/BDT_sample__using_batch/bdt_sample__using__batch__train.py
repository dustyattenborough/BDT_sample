#!/usr/bin/env python

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
sys.path.append("./python")

parser = argparse.ArgumentParser()
## Parameters for the configuration, output, etc
parser.add_argument('-c', '--config', action='store', type=str, default='config.yaml', help='Configration file with sample information')
parser.add_argument('-o', '--output', action='store', type=str, required=True, help='Path to output directory')
## Parameters for the dataset
subrunModes = ('all', 'even', 'odd')
parser.add_argument('--subrun', action='store', type=str, choices=subrunModes, help='Select subruns')
## Parameters for the training
parser.add_argument('--batch', action='store', type=int, help='Batch size')
parser.add_argument('--lr', action='store', type=float, help='Learning rate')
parser.add_argument('--seed', action='store', type=int, help='Random seed')
parser.add_argument('--shuffle', action='store', type=bool, help='Turn on shuffle')
## Parameters for the computing resource
parser.add_argument('--nthread', action='store', type=int, default=os.cpu_count(), help='Number of threads for main')
parser.add_argument('--nloader', action='store', type=int, default=min(8, os.cpu_count()), help='Number of dataLoaders')
## Other parameters
parser.add_argument('-q', '--no-progress-bar', action='store_true', help='Hide progress bar')
args = parser.parse_args()

from utils.Config import Config, overrideConfig
import yaml
config = Config(yaml.load(open(args.config).read(), Loader=yaml.FullLoader))

## Override options
overrideConfig(config, 'dataset/subrun', args, 'subrun')
overrideConfig(config, 'training/batch', args, 'batch', astype=int)
overrideConfig(config, 'training/learningRate', args, 'lr', astype=float)
overrideConfig(config, 'training/seed', args, 'seed', astype=int)
overrideConfig(config, 'training/shuffle', args, 'shuffle', astype=bool)

import numpy as np
import torch
np.seterr(divide='ignore', invalid='ignore')
from dataset.WFDataset import WFDataset as WFDataset

if not os.path.exists(args.output):
    os.makedirs(args.output)

with open(os.path.join(args.output, "config.yaml"), 'w') as fout:
    yaml.dump(config.data, fout)

if args.seed > 0: torch.manual_seed(args.seed)

##### Define dataset instance #####
import pandas as pd
dset = WFDataset(subrun=args.subrun)
df = pd.DataFrame(yaml.load(open("config_dataset/datasets.yaml"), Loader=yaml.FullLoader)['datasets'])
df = df.query(config['dataset/query_train'])
for label, paths in df[["label", "paths"]].to_numpy():
    for path in paths:
        print(label, path)
        dset.addSample(label, path)
dset.initialize()
print(dset)

##### Define dataset instance #####
lengths = [int(x*len(dset)) for x in config['dataset/splitFractions']]
lengths[-1] = len(dset) - sum(lengths[:-1])
if len(lengths) == 2: lengths.append(0)
trnDset, valDset, testDset = torch.utils.data.random_split(dset, lengths)

from torch.utils.data import DataLoader
kwargs = {'num_workers':args.nloader, 'pin_memory':True}
trnLoader = DataLoader(trnDset, batch_size=args.batch, shuffle=args.shuffle, **kwargs)
valLoader = DataLoader(valDset, batch_size=args.batch, shuffle=True, **kwargs)

device = 'cpu'

##### Start training #####
with open(args.output+'/summary.txt', 'w') as fout:
    fout.write(str(args))
    fout.write('\n\n')
    fout.close()

import csv
from sklearn.metrics import accuracy_score
if not args.no_progress_bar:
    from tqdm import tqdm
else:
    tqdm = lambda x, **kwargs: x

boosted_model = None

import math
import time
start = time.time()

print("start !")

val_data = np.zeros([500000,96*208])
val_label = np.ones([500000])*-1

nbatch = args.batch
for i, (data_val, weight_val, fracQ_val, label_val) in enumerate(tqdm(valLoader)):
    nbatch2 = nbatch
    
    #if i==0:
    #    print("fracQ : ", fracQ_val)  
    #    print(  np.array(fracQ_val).shape ) 

    print("i : ",i)
    #if i==20: break
    data_val = data_val.float().numpy()
    
    #print("A")
    if nbatch != len(label_val): 
        nbatch2 = len(label_val)
    #print("B")
    val_data[i*nbatch:(i*nbatch+nbatch2)] += data_val[:,:,40:].reshape((len(data_val),96*208)) 
    val_label[i*nbatch:(i*nbatch+nbatch2)] = label_val.float().numpy()
    #print("C")
val_data = val_data[:-len(val_label[val_label==-1])]
val_label = val_label[:-len(val_label[val_label==-1])]

end = time.time()
print( "store time : " )
print(f"{end - start:.5f} sec")
print("# of validation event : ", len(val_label))
print("\n")    
evals_result = {}

start = time.time()
print("start training !")


train_loss = np.ones([300000])*-1
val_loss = np.ones([300000])*-1


max_boost=1000
iter_boost=100 
N_early_stop=10

for ibatch, (data, weight, fracQ, label) in enumerate(trnLoader):
    print("ibatch : ",ibatch)
    data = data.float().numpy()
    #print( "data: ",data.shape)
    data = data[:,:,40:].reshape((len(data),96*208))
    label = label.float().numpy()
    weight = weight.float().numpy()



    if ibatch==0:
        evals = [(data,label),(val_data,val_label)]

        bojung = np.sort(np.unique(weight))
        pos_weight = bojung[1]/bojung[0]
        print("pos_weight : ",pos_weight)

        boosted_model = xgb.XGBClassifier(
            max_depth=4,
            n_estimators=10, 
            learning_rate=args.lr,
            random_state=args.seed,
            subsample=0.5,
            #colsample_bytree=0.8, 
            eval_metric="logloss",
            scale_pos_weight=pos_weight,
            early_stopping_rounds=N_early_stop
        )
        boosted_model.fit(data, label, eval_set=evals,verbose=True)

    else:
        if  args.batch == len(label):
            if boosted_model.n_estimators <max_boost: boosted_model.n_estimators += iter_boost
            evals = [(data,label),(val_data,val_label)]
            boosted_model.fit(data, label,eval_set=evals,verbose=True,\
                               xgb_model=boosted_model.get_booster())
                               
        else:
            evals = [(data,label),(val_data,val_label)]
            if boosted_model.n_estimators <max_boost: boosted_model.n_estimators += iter_boost
            if ibatch%10==0: boosted_model.fit(data, label, eval_set=evals,verbose=True,\
                                               xgb_model=boosted_model.get_booster())
            else: 
                boosted_model.fit(data, label,eval_set=evals,verbose=True,\
                                   xgb_model=boosted_model.get_booster())
                evals_result  = boosted_model.evals_result()


    print("n_estimator : ", boosted_model.n_estimators)    

    end = time.time()
    print( "batch time : " )
    print(f"{end - start:.5f} sec")


boosted_model.save_model("incremental_xgb_model_4.json")

end = time.time()
print( "training time : " )
print(f"{end - start:.5f} sec")

train_logloss = evals_result["validation_0"]["logloss"]
val_logloss = evals_result["validation_1"]["logloss"]

plt.figure(figsize=(8, 5))
plt.plot(train_logloss, label="Train LogLoss", color="blue")
plt.plot(val_logloss, label="Validation LogLoss", color="red")
plt.xlabel("Boosting Rounds")
plt.ylabel("LogLoss")
plt.title("Train vs Validation LogLoss")
plt.legend()
plt.show()


