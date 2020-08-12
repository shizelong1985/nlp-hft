import pandas as pd
from models.helpers import UtilMat

from models.recommender_systems import LatentFactor
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split

df = pd.read_csv('../data/ratings_shuffled.csv')

# Data preparation (split into 8:1:1)
training_data, temp = train_test_split(df, train_size=0.8, stratify=df[['userid', 'movieid']])
validation_data, test_data = train_test_split(temp, train_size=0.5, stratify=temp[['userid', 'movieid']])

# Using latent factor model for prediction
lf = LatentFactor(K=5, lamb=0.1, verbose=True)

# Training the model
lf.fit(training_data, validation_data=validation_data)

loss = lf.history['loss']
val_loss = lf.history['val_loss']
l = len(loss)

# Plotting together
fig, ax = plt.subplots()
ax.plot(np.arange(0, l), loss, color='red', label='Training loss')
# ax.plot(np.arange(0, l), val_loss, color='blue', label='Validation loss')
ax.set_xlabel('Iterations')
ax.set_ylabel('MSE')
ax.set_title('Loss curve')
ax.legend()
