#!/usr/bin/env python
import torch
import h5py
from tqdm import tqdm
from glob import glob
import numpy as np

## Dataset for the combined waveform dataset converted to hdf5
class WFDataset(torch.utils.data.Dataset):
    def __init__(self, **kwargs):
        super(WFDataset, self).__init__()

        self.isInit = False

        self.opt_subrun = 'all' if 'subrun' not in kwargs else kwargs['subrun']
        self.shape = None
        self.nEventsTotal = 0

        self.fNames = []
        self.labels = []
        self.nEvents = [] ## number of events in each file
        self.cEvents = None ## cumulative sum of nEvents in each file

        self.fracQs = []
        self.sumQs = []

        self.sumWByLabel = torch.tensor([0., 0.], dtype=torch.float32)
        self.minSumW = 1

    def __len__(self):
        return self.nEventsTotal
    
    def __str__(self):
        s = [
            '---- Dataset ----',
            f'* select subruns by: {self.opt_subrun}',
            f'* shape   = {self.shape}',
            f'* nFiles  = {len(self.fNames)}',
            f'* nEvents = {self.nEventsTotal}',
            f'* ME(label=1) sumW = {self.sumWByLabel[1]:.3f}',
            f'* FN(label=0) sumW = {self.sumWByLabel[0]:.3f}',
        ]
        return '\n'.join(s)
    
    def __getitem__(self, idx):
        if not self.isInit: self.initialize()
        fileIdx = torch.searchsorted(self.cEvents, idx)
        ii = idx-self.cEvents[fileIdx]

        label = self.labels[fileIdx]

        fName = self.fNames[fileIdx]
        fin = h5py.File(fName, 'r', libver='latest', swmr=True)

        wf = fin['events/waveform'][ii]
        nch, nT = self.shape

        wfmask = wf > -1e9 ## combinedWF code fills up padding with -9e9
        wf /= wf.max(axis=1)[:,np.newaxis] ## Scale by peak, to be 1.0
        np.nan_to_num(wf, copy=False, nan=0.0)
        wf = torch.tensor(wf*wfmask, dtype=torch.float32)

        fracQ = self.fracQs[fileIdx][ii]

        eventWeight = self.minSumW/self.sumWByLabel[label]

        return wf, eventWeight, fracQ, label

    def initialize(self):
        assert(self.isInit == False)

        ## Convert internal data to torch tensors
        self.labels = torch.tensor(self.labels, dtype=torch.int8)
        self.nEvents = torch.tensor(self.nEvents, dtype=torch.int32)
        self.cEvents = self.nEvents.cumsum(dim=0)
        self.nEventsTotal = self.cEvents[-1]

        ## Calculate sum of weights, to scale down majorities
        for label, sumQ in zip(self.labels, self.sumQs):
            self.sumWByLabel[label] += len(sumQ)
        self.minSumW = self.sumWByLabel.min()
        assert(self.minSumW > 0)

        self.isInit = True

    def addSample(self, label, fNamePattern):
        labelIdx = ['FN', 'ME'].index(label)

        for fName in tqdm(glob(fNamePattern), "Loading %s" % label):
            if not fName.endswith('.h5'): continue

            ## Check the fileName fits into the JSNS2 file naming rule
            _, run, subrun, ext = fName.rsplit('.', 3)
            if not run.startswith('r') or not subrun.startswith('f') or ext != 'h5':
                print(f"Filename does not fit to the '.r######.f######.h5' format. Skip this file. {fName}")
                continue
            ## Keep only even subruns or odd subruns to split training/validation/test sets 
            ## if the subrun-splitting option is enabled
            subrun = subrun[1:].lstrip('0')
            subrun = int(subrun) if len(subrun) > 0 else 0
            if self.opt_subrun == 'odd' and (subrun % 2 != 1): continue
            elif self.opt_subrun == 'even' and (subrun % 2 != 0): continue

            ## Check the shape of the input file
            fin = h5py.File(fName, 'r', libver='latest', swmr=True)
            shape = fin['info/shape'][:]
            if self.shape is None:
                self.shape = shape
            elif np.abs(self.shape - shape).sum() != 0:
                print(f'\n!!! Inconsistent data shape. Skip this file {fName}')
                continue

            ## Keep this file
            self.fNames.append(fName)
            self.labels.append(labelIdx)
            nEvent = len(fin['events/trigID'])
            self.nEvents.append(nEvent)

            ## Sum of the charge has to be retrieves before the main loop 
            ## because we need total sumQ of full samples
            ## On the other hand, the waveform could not be loaded here
            ## because it is too heavy
            pmtQ = fin['events/pmtQ'][()]

            validPMT = fin['events/validPMT'][()]
            pmtQ *= validPMT
            sumQ = pmtQ.sum(axis=1)
            sumQ = torch.tensor(sumQ, dtype=torch.float32)
            self.sumQs.append(sumQ)

            fracQ = pmtQ/sumQ[:, np.newaxis]
            fracQ = np.nan_to_num( fracQ, nan=0 )
            fracQ = torch.tensor(fracQ, dtype=torch.float32)
            self.fracQs.append(fracQ)
