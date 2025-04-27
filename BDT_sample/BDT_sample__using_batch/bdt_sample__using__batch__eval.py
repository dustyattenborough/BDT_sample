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
## Parameters for the training results, input samples and evaluation output
parser.add_argument('-t', '--train', action='store', type=str, required=True, help='Path to training results directory')
parser.add_argument('-o', '--output', action='store', type=str, required=True, help='Path to output directory')
parser.add_argument('-c', '--config', action='store', type=str, required=False, help='Configration file with sample information')
parser.add_argument('-f', '--filename', action='store', type=str, required=True, help='json file which trained')
## Parameters for the dataset
subrunModes = ('all', 'even', 'odd')
parser.add_argument('--subrun', action='store', type=str, choices=subrunModes, help='Select subruns')
parser.add_argument('--batch', action='store', type=int, default=256, help='Batch size')
## Parameters for the computing resource
parser.add_argument('--nthread', action='store', type=int, default=os.cpu_count(), help='Number of threads for main')
parser.add_argument('--nloader', action='store', type=int, default=min(8, os.cpu_count()), help='Number of dataLoaders')
parser.add_argument('--device', action='store', type=int, default=-1, help='Device number (-1 for CPU)')
args = parser.parse_args()

from utils.Config import Config, overrideConfig
import yaml
config = args.config or args.train+'/config.yaml'
config = Config(yaml.load(open(config).read(), Loader=yaml.FullLoader))

## Override options
config['dataset/subrun'] = args.subrun

if os.path.exists(args.output):
    print("Warning: Output directory already exists, ", args.output)
    print("         we overwrite score file - hope this is safe")
else:
    os.makedirs(args.output)


import numpy as np
import torch
np.seterr(divide='ignore', invalid='ignore')
from dataset.WFDataset import WFDataset as WFDataset

if not os.path.exists(args.output):
    os.makedirs(args.output)

with open(os.path.join(args.output, "config.yaml"), 'w') as fout:
    yaml.dump(config.data, fout)


##### Define dataset instance #####
import pandas as pd
dset = WFDataset(subrun=args.subrun)
df = pd.DataFrame(yaml.load(open("config_dataset/datasets.yaml"), Loader=yaml.FullLoader)['datasets'])
df = df.query(config['dataset/query_eval'])
for label, paths in df[["label", "paths"]].to_numpy():
    for path in paths:
        print(label, path)
        dset.addSample(label, path)
dset.initialize()
print(dset)

##### Define dataset instance #####
from torch.utils.data import DataLoader
kwargs = {'num_workers':args.nloader, 'pin_memory':True}
testLoader = DataLoader(dset, batch_size=args.batch, shuffle=False, **kwargs)


device = 'cpu'

import csv
from sklearn.metrics import accuracy_score
from tqdm import tqdm

boosted_model = None

import math
import time

print("start !")

start = time.time()
print("start training !")

boosted_model = xgb.XGBClassifier()
boosted_model.load_model( args.filename )

eval_pred = np.zeros([550000])
eval_label = np.ones([550000])*-1


max_boost=500 # 1000
iter_boost=5 # 10
nbatch = args.batch
for ibatch, (data, weight, fracQ, label) in enumerate(tqdm(testLoader)):
    nbatch2 = nbatch
    if nbatch != len(label):
        nbatch2 = len(label)
    #print("ibatch : ",ibatch)
    data = data.float().numpy()
    #print( "data: ",data.shape)
    data = data[:,:,40:].reshape((len(data),96*208))
    label = label.float().numpy()
    weight = weight.float().numpy()

    eval_pred[ibatch*nbatch:(ibatch*nbatch+nbatch2)] += np.array(boosted_model.predict_proba(data)[:,1])
    eval_label[ibatch*nbatch:(ibatch*nbatch+nbatch2)] = np.array(label)
    
    #break
end = time.time()
print( "batch time : " )
print(f"{end - start:.5f} sec")

eval_pred = eval_pred[:-len(eval_label[eval_label==-1])]
eval_label = eval_label[:-len(eval_label[eval_label==-1])]
eventWeight = eval_label * ( np.sum(eval_label<1)/len(eval_label) ) + np.abs(1-eval_label)


import pandas as pd
df = pd.DataFrame({"pred":eval_pred, "label":eval_label, "weight":eventWeight})
df.to_csv(args.filename.split(".")[0]+".csv")

sig = np.array(eval_pred[(eval_label==1)])
bkg = np.array(eval_pred[(eval_label==0)])

print("sigmax = ",np.max(sig))
print("bkgmax = ",np.max(bkg))
print( np.sum(sig==1))
print("sigmin = ",np.min(sig))
print("bkgmin = ",np.min(bkg))

fn = np.array(bkg)
me = np.array(sig)

print(me)
print(me.shape)

print(fn)
print(fn.shape)
fn.sort()
me.sort()


fn_rej_list = np.array([ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.96, 0.97, 0.98,0.99 ])
me_eff_list = np.array([len(me[me>fn[int(ii*len(fn))]])/len(me) for ii in fn_rej_list  ])

plt.plot(100*fn_rej_list, 100*me_eff_list, "*-")
plt.xlabel("fn_rej (%)")
plt.ylabel("me_eff (%)")
plt.ylim(90,100)
plt.grid()
plt.show()

fn_99 = fn[int(0.99*len(fn))]
print("\n\n")
#print("\n\n 99 point : %.5f"%fn_99)
print("ME eff: %.3f , FN rej: %.3f"%(100*len(me[me>fn_99])/len(me), 100*len(fn[fn<fn_99])/len(fn) ))


fn_95 = fn[int(0.95*len(fn))]
#print("\n\n 95 point : %.5f"%fn_95)
print("ME eff: %.3f , FN rej: %.3f"%(100*len(me[me>fn_95])/len(me), 100*len(fn[fn<fn_95])/len(fn) ))

fn_90 = fn[int(0.90*len(fn))]
#print("\n\n 90 point : %.5f"%fn_90)
print("ME eff: %.3f , FN rej: %.3f"%(100*len(me[me>fn_90])/len(me), 100*len(fn[fn<fn_90])/len(fn) ))


print("\n\n <-pred0.5-> : ")
print("ME eff: %.3f , FN rej: %.3f"%(100*len(me[me>0.5])/len(me), 100*len(fn[fn<0.5])/len(fn)))


plt.hist(sig, histtype="step",weights=eventWeight[(eval_label==1)]/np.sum( eventWeight[(eval_label==1)] ), bins=np.linspace(0,1,50),alpha=0.7,color="blue",label="ME")
plt.hist(bkg, histtype="step",weights=eventWeight[(eval_label==0)]/np.sum( eventWeight[(eval_label==0)] ), bins=np.linspace(0,1,50),alpha=0.7,color="red",label="FN")
plt.ylim([0,1])
plt.ylabel('A.U.')
plt.xlabel('Pred')
plt.legend()
plt.grid()
plt.show()

