# Mega Case Study -> Make a Hybrid Deep Learning Model

# Part 1 -> Fraud detection with Self-Organized Map

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importing the dataset
dataset = pd.read_csv("data/Credit_Card_Applications.csv" )
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

#Feature scalling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
x = sc.fit_transform(x)

#Training the SOM
from minisom import MiniSom
som = MiniSom(x = 10, y = 10, input_len = 15, sigma=1.0, learning_rate=0.5)
som.random_weights_init(x)
som.train_random(data = x, num_iteration = 100)

#Visualing 
from pylab import bone,pcolor,colorbar,plot,show
bone()
pcolor(som.distance_map().T)
colorbar()
markers = ["o","s"]
colors = ["r","g"]
for i,X in enumerate(x):
    w = som.winner(X)
    plot(w[0] + 0.5,
         w[1] + 0.5,
         markers[y[i]],
         markeredgecolor = colors[y[i]],
         markerfacecolor = "None",
         markersize = 10,
         markeredgewidth = 2)
show()
mappings = som.win_map(x)
frauds = np.concatenate((mappings[(5,3)],mappings[(8,3)]),axis=0)
frauds = sc.inverse_transform(frauds)



# Part 2 -> Going from Unsupervised to Supervised Deep Learning

# Creating the matrix of features
customers = dataset.iloc[:,1:].values

# Creating the dependent variable
is_fraud = np.zeros(len(dataset))
for i in range(len(dataset)):
    if dataset.iloc[i,0] in frauds :
        is_fraud[i] = 1

from sklearn.preprocessing import StandardScaler
sts = StandardScaler()
customers = sts.fit_transform(customers)

from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()

classifier.add(Dense(units = 2,kernel_initializer="uniform",activation="relu",input_dim = 15))
classifier.add(Dense(units = 1,kernel_initializer="uniform",activation="sigmoid"))

classifier.compile(optimizer = "adam", loss = "binary_crossentropy", metrics=["accuracy"])
classifier.fit(customers,is_fraud,batch_size=1,epochs=2)

y_pred = classifier.predict(customers)
y_pred = np.concatenate((dataset.iloc[:,0:1], y_pred),axis=1)
y_pred = y_pred[y_pred[:,1].argsort()]















