import pandas as pd
from matplotlib import pyplot as plt

scores = pd.read_csv('logtest.csv')

print(scores.columns)

plt.plot(scores['loss'])
plt.plot(scores['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()