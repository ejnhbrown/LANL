import pickle
import xgboost as xgb
import matplotlib.pyplot as plt

model = pickle.load(open('benchmarkcnnII.pkl.dat','rb'))
fig, ax = plt.subplots(figsize=(12,18))
xgb.plot_importance(model, max_num_features=200, height=0.8, ax=ax)
plt.show()