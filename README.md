# BDT_sample
## BDT sample simple
basic BDT example
using array ->  (eventN x pulse bin)
    
### Data split
```
X_train, X_test, y_train, y_test = train_test_split(wf_all_1563, label_all_1563, random_state=args.random_states,test_size=args.test_size)
```
### training
```
boosted_model = xgb.XGBClassifier(max_depth=args.depth, n_estimators=args.estimators, learning_rate=args.lr, random_state=args.random_states, subsample=0.5, eval_metric='logloss', scale_pos_weight = pos_weight)#, early_stopping_rounds=50)
boosted_model.fit(X_train, y_train, eval_set=evals, verbose=True )#, eval_metric="log_loss")
```
### evaluation
```
y_pred_proba_1592 = boosted_model.predict_proba(wf_all_1592)
```
    

## BDT sample using batch
using batch like cnn
all event is spliting 
By training with small batches, even very large datasets can be handled without issue.

training code : BDT_sample/BDT_sample__using_batch/bdt_sample__using__batch__train.py
evaluation code : BDT_sample/BDT_sample__using_batch/bdt_sample__using__batch__eval.py

### make dataset 
```
##### Define dataset instance #####
lengths = [int(x*len(dset)) for x in config['dataset/splitFractions']]
lengths[-1] = len(dset) - sum(lengths[:-1])
if len(lengths) == 2: lengths.append(0)
trnDset, valDset, testDset = torch.utils.data.random_split(dset, lengths)

from torch.utils.data import DataLoader
kwargs = {'num_workers':args.nloader, 'pin_memory':True}
trnLoader = DataLoader(trnDset, batch_size=args.batch, shuffle=args.shuffle, **kwargs)
valLoader = DataLoader(valDset, batch_size=args.batch, shuffle=True, **kwargs)
```
### training
```
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
```
### evaluation
```
    eval_pred[ibatch*nbatch:(ibatch*nbatch+nbatch2)] += np.array(boosted_model.predict_proba(data)[:,1])
```
