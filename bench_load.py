import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import time
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from pickle import dump

X_train_scaled = pd.read_pickle('Xtrain_bench.pkl')
X_test_scaled = pd.read_pickle('Xtest_bench.pkl')
y_tr = pd.read_pickle('y_train.pkl')

n_fold = 5
folds = KFold(n_splits=n_fold, shuffle=True, random_state=11)

print(y_tr.shape)

def train_model(X=X_train_scaled, X_test=X_test_scaled, y=y_tr, params=None, folds=folds, model_type='lgb'):
    oof = np.zeros(len(X))
    prediction = np.zeros(len(X_test))
    scores = []

    for fold_n, (train_index, valid_index) in enumerate(folds.split(X)):
        print('Fold', fold_n+1, 'started at', time.ctime())
        X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

        if model_type == 'xgb':
            train_data = xgb.DMatrix(data=X_train, label=y_train, feature_names=X.columns)
            valid_data = xgb.DMatrix(data=X_valid, label=y_valid, feature_names=X.columns)

            watchlist = [(train_data, 'train'), (valid_data, 'valid_data')]
            model = xgb.train(dtrain=train_data, num_boost_round=20000, evals=watchlist, early_stopping_rounds=500,
                              verbose_eval=500, params=params)
            y_pred_valid = model.predict(xgb.DMatrix(X_valid, feature_names=X.columns),
                                         ntree_limit=model.best_ntree_limit)
            y_pred = model.predict(xgb.DMatrix(X_test, feature_names=X.columns), ntree_limit=model.best_ntree_limit)

        oof[valid_index] = y_pred_valid.reshape(-1, )
        scores.append(mean_absolute_error(y_valid, y_pred_valid))

        prediction += y_pred

    prediction /= n_fold

    print('CV mean score: {0:.4f}, std: {1:.4f}.'.format(np.mean(scores), np.std(scores)))
    return oof, prediction, model


xgb_params = {'eta': 0.05, 'max_depth': 10, 'subsample': 0.9, #'colsample_bytree': 0.8,
          'objective': 'reg:linear', 'eval_metric': 'mae', 'silent': True, 'nthread': 4}
oof_xgb, prediction_xgb, model = train_model(params=xgb_params, model_type='xgb')
dump(model,open('benchmark.pkl.dat', 'wb'))

submission = pd.read_csv('sample_submission.csv')

print(prediction_xgb.shape)

submission.iloc[:,1] = prediction_xgb
submission.to_csv('benchmarksubmission.csv',index=False)

plt.figure(figsize=(16, 8))
plt.plot(y_tr, color='g', label='y_train')
plt.plot(oof_xgb, color='teal', label='xgb')
plt.legend();
plt.title('Predictions vs actual');
plt.show()
