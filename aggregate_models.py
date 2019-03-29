import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import random

import xgboost as xgb
import lightgbm as lgb
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import NuSVR
from catboost import CatBoostRegressor

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from pickle import dump


# Import data:

X_train_scaled = pd.read_pickle('Xtrain_benchII.pkl')
X_test_scaled = pd.read_pickle('Xtest_benchII.pkl')
y_tr = pd.read_pickle('y_trainII.pkl')
print(X_train_scaled.shape)

# Set up folds:
n_fold = 5
folds = KFold(n_splits=n_fold, shuffle=True, random_state=11)


# load CNN/2nd stage params (optional):
# Set cnn as: 'first' to load minichunk cnn data,
# 'second' to load chunk cnn data,
# 'none' to load no cnn data and just run on aggregate data.

cnn = 'none'    # CHANGE THIS

if cnn == 'second':
    print('adding first stage cnn features')
    direc = 'chunk_data/'
    columns = ['cnn'+str(i) for i in range(1000)]
    index = [i for i in range(4194)]
    X_cnn = pd.DataFrame(index=index, columns=columns, dtype='float64')

    for i in range(4194):
        for j in range(10):
            target_dir = direc + str(i)
            target_dir = target_dir + '/' + str(j) + 'n.npy'
            X = np.load(target_dir)
            cols = ['cnn'+str(k+(j*100)) for k in range(100)]
            for l, col in enumerate(cols):
                X_cnn.at[i, col] = X[0, l]

    direc = 'test_extract/test_chunk_data/'
    index = [i for i in range(2624)]
    Xte_cnn = pd.DataFrame(index=index, columns=columns, dtype='float64')

    for i in range(2624):
        for j in range(10):
            target_dir = direc + str(i)
            target_dir = target_dir + '/' + str(j) + 'n.npy'
            X = np.load(target_dir)
            cols = ['cnn' + str(k + (j * 100)) for k in range(100)]
            for l, col in enumerate(cols):
                Xte_cnn.at[i, col] = X[0, l]

    X_train_scaled = pd.concat([X_train_scaled, X_cnn], axis=1)
    X_test_scaled = pd.concat([X_test_scaled, Xte_cnn], axis=1)

elif cnn == 'second':
    print('adding second stage cnn features')
    direc = 'full_chunk_data/'
    columns = ['cnn' + str(i) for i in range(100)]
    index = [i for i in range(4194)]
    X_cnn = pd.DataFrame(index=index, columns=columns, dtype='float64')

    for i in range(4194):
        target_dir = direc + str(i) + '.npy'
        X = np.load(target_dir)
        for l, col in enumerate(columns):
            X_cnn.at[i, col] = X[0, l]

    direc = 'test_extract/test_full_chunk_data/'
    index = [i for i in range(2624)]
    Xte_cnn = pd.DataFrame(index=index, columns=columns, dtype='float64')

    for i in range(2624):
        target_dir = direc + str(i) + '.npy'
        X = np.load(target_dir)
        for l, col in enumerate(columns):
            Xte_cnn.at[i, col] = X[0, l]

    X_train_scaled = pd.concat([X_train_scaled, X_cnn], axis=1)
    X_test_scaled = pd.concat([X_test_scaled, Xte_cnn], axis=1)

elif cnn == 'none':
    print('no cnn features added')

# Training the model: this is called several times, and runs differently depending
# on the model or params passed.

print('defining models')
def train_model(X=X_train_scaled, X_test=X_test_scaled, y=y_tr, params=None, folds=folds, model_type='lgb', model=None):
    oof = np.zeros(len(X))
    prediction = np.zeros(len(X_test))
    scores = []

    for fold_n, (train_index, valid_index) in enumerate(folds.split(X)):
        print('Fold', fold_n+1, 'started at', time.ctime())
        X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

        if model_type == 'lgb':
            model = lgb.LGBMRegressor(**params, n_estimators=50000, n_jobs=-1)
            model.fit(X_train, y_train,
                      eval_set=[(X_train, y_train), (X_valid, y_valid)], eval_metric='mae',
                      verbose=10000, early_stopping_rounds=200)

            y_pred_valid = model.predict(X_valid)
            y_pred = model.predict(X_test, num_iteration=model.best_iteration_)

        if model_type == 'xgb':
            train_data = xgb.DMatrix(data=X_train, label=y_train, feature_names=X.columns)
            valid_data = xgb.DMatrix(data=X_valid, label=y_valid, feature_names=X.columns)

            watchlist = [(train_data, 'train'), (valid_data, 'valid_data')]
            model = xgb.train(dtrain=train_data, num_boost_round=20000, evals=watchlist, early_stopping_rounds=200,
                              verbose_eval=500, params=params)
            y_pred_valid = model.predict(xgb.DMatrix(X_valid, feature_names=X.columns),
                                         ntree_limit=model.best_ntree_limit)
            y_pred = model.predict(xgb.DMatrix(X_test, feature_names=X.columns), ntree_limit=model.best_ntree_limit)

        if model_type == 'sklearn':
            model = model
            model.fit(X_train, y_train)

            y_pred_valid = model.predict(X_valid).reshape(-1, )
            score = mean_absolute_error(y_valid, y_pred_valid)
            print(f'Fold {fold_n}. MAE: {score:.4f}.')
            print('')

            y_pred = model.predict(X_test).reshape(-1, )

        # if model_type == 'cat':
        #     model = CatBoostRegressor(iterations=20000, task_type='GPU', eval_metric='MAE', **params)
        #     model.fit(X_train, y_train, eval_set=(X_valid, y_valid), cat_features=[], use_best_model=True,
        #               verbose=False)
        #
        #     y_pred_valid = model.predict(X_valid)
        #     y_pred = model.predict(X_test)

        oof[valid_index] = y_pred_valid.reshape(-1, )
        scores.append(mean_absolute_error(y_valid, y_pred_valid))

        prediction += y_pred

    prediction /= n_fold

    print('CV mean score: {0:.4f}, std: {1:.4f}.'.format(np.mean(scores), np.std(scores)))
    CVscore = np.mean(scores)

    return oof, prediction, CVscore


# define the parameters and models for the training function:

lgb_params = {'num_leaves': 54,
              'min_data_in_leaf': 79,
              'objective': 'huber',
              'max_depth': -1,
              'learning_rate': 0.01,
              "boosting": "gbdt",
              "bagging_freq": 5,
              "bagging_fraction": 0.8126672064208567,
              "bagging_seed": 11,
              "metric": 'mae',
              "verbosity": -1,
              'reg_alpha': 0.1302650970728192,
              'reg_lambda': 0.3603427518866501
             }

xgb_params = {'eta': 0.03,
              'max_depth': 10,
              'subsample': 0.9,
              'objective': 'reg:linear',
              'eval_metric': 'mae',
              'silent': True,
              'nthread': 4
              }

NuSVR_model = NuSVR(gamma='scale', nu=0.9, C=10.0, tol=0.01)
NuSVR2_model = NuSVR(gamma='scale', nu=0.7, tol=0.01, C=1.0)
cat_params = {'loss_function':'MAE'}
KR_model = KernelRidge(kernel='rbf', alpha=0.1, gamma=0.01)

# Run the models:
print('running models')
print('xgb:')
oof_xgb, prediction_xgb, CVscore_xgb = train_model(params=xgb_params, model_type='xgb')
print('lgb:')
oof_lgb, prediction_lgb, CVscore_lgb = train_model(X=X_train_scaled, X_test=X_test_scaled, params=lgb_params, model_type='lgb')
print('svr:')
oof_svr, prediction_svr, CVscore_svr = train_model(X=X_train_scaled, X_test=X_test_scaled, params=None, model_type='sklearn', model=NuSVR_model)
print('svr2:')
oof_svr2, prediction_svr2, CVscore_svr2 = train_model(X=X_train_scaled, X_test=X_test_scaled, params=None, model_type='sklearn', model=NuSVR2_model)
# print('cat:') # NOTE: disabled for now, as cuda version required for gpu acceleration
# is incompatible with cuda version required for tf_gpu. If re-enabled, re-add to blend variables
# oof_cat, prediction_cat, CVscore_cat = train_model(X=X_train_scaled, X_test=X_test_scaled, params=cat_params, model_type='cat')
print('kr:')
oof_kr, prediction_kr, CVscore_kr = train_model(X=X_train_scaled, X_test=X_test_scaled, params=None, model_type='sklearn', model=KR_model)

# blend:
oof_blend = (oof_xgb + oof_lgb + oof_svr + oof_svr2 + oof_kr) / 6
prediction_blend = (prediction_xgb + prediction_lgb + prediction_svr + prediction_svr2
                    + prediction_kr) / 6

# compute blended CVscore:
CVscore_blend = 0
for i in range(4194):
    CVscore_blend += abs(oof_blend[i]-float(y_tr.iloc[i]))
CVscore_blend /= int(y_tr.shape[0])

# save submission files:
print('submitting csvs')
submission = pd.read_csv('sample_submission.csv')
submission.iloc[:,1] = prediction_lgb
submission.to_csv(f'aggregate_predictions/agg-lgb_{CVscore_lgb}_cnnfeatures-{cnn}.csv',index=False)
submission.iloc[:,1] = prediction_xgb
submission.to_csv(f'aggregate_predictions/agg-xgb_{CVscore_xgb}_cnnfeatures-{cnn}.csv',index=False)
submission.iloc[:,1] = prediction_svr
submission.to_csv(f'aggregate_predictions/agg-svr_{CVscore_svr}_cnnfeatures-{cnn}.csv',index=False)
submission.iloc[:,1] = prediction_svr2
submission.to_csv(f'aggregate_predictions/agg-svr2_{CVscore_svr2}_cnnfeatures-{cnn}.csv',index=False)
# submission.iloc[:,1] = prediction_cat
# submission.to_csv(f'aggregate_predictions/agg-cat_{CVscore_cat}_cnnfeatures-{cnn}.csv',index=False)
submission.iloc[:,1] = prediction_kr
submission.to_csv(f'aggregate_predictions/agg-kr_{CVscore_kr}_cnnfeatures-{cnn}.csv',index=False)
submission.iloc[:,1] = prediction_blend
submission.to_csv(f'aggregate_predictions/agg-blend_{CVscore_blend}_cnnfeatures-{cnn}.csv',index=False)

# plot predictions vs actual:
plt.figure(figsize=(18, 12))
plt.subplot(2, 3, 1)
plt.plot(y_tr, color='g', label='y_train')
plt.plot(oof_lgb, color='b', label='lgb')
plt.legend(loc=(1, 0.5));
plt.title('lgb: '+str(CVscore_lgb));
plt.subplot(2, 3, 2)
plt.plot(y_tr, color='g', label='y_train')
plt.plot(oof_xgb, color='teal', label='xgb')
plt.legend(loc=(1, 0.5));
plt.title('xgb: '+str(CVscore_xgb));
plt.subplot(2, 3, 3)
plt.plot(y_tr, color='g', label='y_train')
plt.plot(oof_svr, color='red', label='svr')
plt.legend(loc=(1, 0.5));
plt.title('svr: '+str(CVscore_svr));
plt.subplot(2, 3, 4)
plt.plot(y_tr, color='g', label='y_train')
plt.plot(oof_svr2, color='red', label='svr2')
plt.legend(loc=(1, 0.5));
plt.title('svr2: '+str(CVscore_svr2));
# plt.subplot(3, 3, 5)
# plt.plot(y_tr, color='g', label='y_train')
# plt.plot(oof_cat, color='b', label='cat')
# plt.legend(loc=(1, 0.5));
# plt.title('cat: '+str(CVscore_cat));
plt.subplot(2, 3, 5)
plt.plot(y_tr, color='g', label='y_train')
plt.plot(oof_kr, color='b', label='kr')
plt.legend(loc=(1, 0.5));
plt.title('kr: '+str(CVscore_kr));
plt.subplot(2, 3, 6)
plt.plot(y_tr, color='g', label='y_train')
plt.plot(oof_blend, color='gold', label='blend')
plt.legend(loc=(1, 0.5));
plt.title('blend: '+str(CVscore_blend));
plt.legend(loc=(1, 0.5));
plt.suptitle('Predictions vs actual');
plt.show()