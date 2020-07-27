import pandas as pd
from models.utilmat import UtilMat

from models.recommender_systems import LatentFactorModel
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split

df = pd.read_csv('../data/ratings_shuffled.csv')

# Data preparation (split into 8:1:1)
training_data, temp = train_test_split(df, train_size=0.8)
validation_data, test_data = train_test_split(temp, train_size=0.5)

validation_data.reset_index(drop=True, inplace=True)
test_data.reset_index(drop=True, inplace=True)

# Using latent factor model for prediction
lf = LatentFactorModel(n=100, lamb=0.01, verbose=True)

# Training the model
lf.fit(training_data)

loss = lf.history['loss']
# val_loss = lf.history['val_loss']
l = len(loss)

# Plotting together
fig, ax = plt.subplots()
ax.plot(np.arange(0, l), loss, color='red', label='Training loss')
# ax.plot(np.arange(0, l), val_loss, color='blue', label='Validation loss')
ax.set_xlabel('Iterations')
ax.set_ylabel('MSE')
ax.set_title('Loss curve')
ax.legend()

# print('Test Loss: ', lf.calc_loss(test_utilmat, get_mae=True))
