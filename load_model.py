from pickle import load
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import timeit
import time
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from pickle import dump

submission = pd.read_pickle('submission.pkl')
y_tr = pd.read_pickle('y_train.pkl')
X_test = pd.read_pickle('Xtest_bench.pkl')

print(y_tr.head())


model = load(open('benchmark.pkl.dat', 'rb'))

predictions = model.predict(xgb.DMatrix(X_test, feature_names=X_test.columns), ntree_limit=model.best_ntree_limit)

submission['time_to_failure'] = predictions
print(submission.head())
submission.to_csv('submission2.csv')

