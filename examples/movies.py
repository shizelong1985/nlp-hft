import pandas as pd
from models.helpers import UtilMat

from models.recommender_systems import LatentFactorSVD
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split

user_id = 'userid'
item_id = 'movieid'
rating_id = 'rating'

df = pd.read_csv('../data/ratings_shuffled.csv')

# Data preparation (split into 8:1:1)
training_data, temp = train_test_split(df, train_size=0.8, stratify=df[user_id])
validation_data, test_data = train_test_split(temp, train_size=0.5)

# Using latent factor model for prediction
lf = LatentFactorSVD(K=5, learning_rate=0.01, lamb=0.1, verbose=True)

# Training the model
lf.fit(training_data, validation_data=validation_data, iters=10)

train_loss = lf.history['loss']
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

print('Test Loss: ', lf.calc_loss(test_data, pandas=True))
