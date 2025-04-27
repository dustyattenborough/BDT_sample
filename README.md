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
      ```
    ### evaluation
      ```
      boosted_model.fit(X_train, y_train, eval_set=evals, verbose=True )#, eval_metric="log_loss")
      y_pred_proba_1592 = boosted_model.predict_proba(wf_all_1592)
      ```
    

## BDT sample using batch
