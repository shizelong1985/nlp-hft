import pandas as pd
from models.utilmat import UtilMat

from models.recommender_systems import LatentFactor
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split


df = pd.read_csv('../data/ratings_shuffled.csv')

# Data preparation (split into 8:1:1)
training_data, temp = train_test_split(df, train_size=0.8)
validation_data, test_data = train_test_split(temp, train_size=0.5)

training_data.reset_index(drop=True, inplace=True)
validation_data.reset_index(drop=True, inplace=True)
test_data.reset_index(drop=True, inplace=True)

# Convert to utilmat
train_utilmat = UtilMat(training_data)
val_utilmat = UtilMat(validation_data)
test_utilmat = UtilMat(test_data)

# Using latent factor model for prediction
lf = LatentFactor(K=5, learning_rate=0.01, lamb=0.1, verbose=True)

# Training the model
lf.fit(train_utilmat, iters=10, val_utilmat=val_utilmat)


train_loss = lf.history['train_loss']
val_loss = lf.history['val_loss']
l = len(train_loss)

# Plotting together
fig, ax = plt.subplots()
ax.plot(np.arange(0, l), train_loss, color='red', label='Training loss')
ax.plot(np.arange(0, l), val_loss, color='blue', label='Validation loss')
ax.set_xlabel('Iterations')
ax.set_ylabel('MSE')
ax.set_title('Loss curve')
ax.legend()

print('Test Loss: ', lf.calc_loss(test_utilmat, get_mae=True))
