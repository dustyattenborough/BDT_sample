# BDT_sample

##  hyper parameter
*max_depth: Controls the maximum depth of each tree, which helps in preventing overfitting by limiting the complexity.
*n_estimators: Number of boosting rounds, essentially how many trees the model will build.
*learning_rate: Shrinks the contribution of each tree to prevent overfitting. Lower values require more trees to model the data.
*random_state: Ensures the results are reproducible.
*subsample: Randomly samples a subset of the training data to build each tree, which can reduce overfitting.
*colsample_bytree: Specifies the fraction of features to be considered for building each tree. Helps with regularization.
*eval_metric: This is the metric used for evaluating the performance of the model. In this case, it's logloss, which is suitable for binary classification.
*scale_pos_weight: Useful in handling class imbalance by assigning higher weight to the positive class.
*early_stopping_rounds: If the model does not improve over a given number of rounds, it stops training early to prevent overfitting and unnecessary computation.


## BDT sample simple
basic BDT example
using array ->  (eventN x pulse bin)

```
    python bdt_xgboost_sh_detail__JADEv0_noPMTsel.py 
```
    
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
using batch spliting like cnn
using array ->  (eventN x pulse bin)

all event is spliting 
All events were split into batches of the size specified in the config.yaml file.
By training with small batches, even very large datasets can be handled without issue.

training code : BDT_sample/BDT_sample__using_batch/bdt_sample__using__batch__train.py
```
    python ${PWDDir}/bdt_xgboost_sh_detail__JADEv0_noPMTsel.py --depth ${depth} --estimators ${estimator} --lr ${lr} --random_states ${rand} --test_size ${test_size} >> Loss_RS_${rand}__pmtSel_${pmtSel}.txt
```
evaluation code : BDT_sample/BDT_sample__using_batch/bdt_sample__using__batch__eval.py
```
    python bdt_xgboost_sh_detail__JADEv0_noPMTsel__96x208__eval.py -o $DIRNAME -f $DIRNAME/model_bdt__xgboost__depth_${depth}__estimator_1200__lr_${lr}__random_${rs}__test_size_0.250__JADEv0_noPMTsel__96x208_train.json --pmtSel $pmtSel
```
### Data split
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
            max_depth=4,  # Maximum depth of the trees. Controls model complexity.
            n_estimators=10,  # Number of boosting rounds (trees).
            learning_rate=args.lr,  # Step size shrinking to prevent overfitting.
            random_state=args.seed,  # Random seed for reproducibility.
            subsample=0.5,  # Fraction of samples to use for training each tree. Helps with regularization.
            colsample_bytree=0.8,  # Fraction of features to use for each tree.
            eval_metric="logloss",  # Evaluation metric, here it's log loss (useful for classification).
            scale_pos_weight=pos_weight,  # Balances the weight of positive and negative classes for imbalanced datasets.
            early_stopping_rounds=N_early_stop  # Stops training if validation metric does not improve after a certain number of rounds.
        )
        boosted_model.fit(data, label, eval_set=evals,verbose=True)
```
### evaluation
```
    eval_pred[ibatch*nbatch:(ibatch*nbatch+nbatch2)] += np.array(boosted_model.predict_proba(data)[:,1])
```
