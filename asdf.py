import numpy as np
import tensorflow as tf
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
import copy

def batchify(Xs, ys, batches):
    Xs, ys = shuffle(Xs, ys, random_state=0)
    batch_size = int(len(Xs)/batches)
    for i in range(batches):
        yield Xs[batch_size*i: batch_size*(i+1)], ys[batch_size*i: batch_size*(i+1)]

def mnist_dataset():
    raw_data = tf.keras.datasets.mnist.load_data(path='mnist.npz')
    train_X, train_y = raw_data[0]
    test_X, test_y = raw_data[1]

    train_X = train_X.reshape((len(train_X), 784))
    test_X = test_X.reshape((len(test_X), 784))

    train_X = train_X/255.0
    test_X = test_X/255.0

    # train_y = np.zeros((len(train_y_raw), 10))
    # train_y[np.arange(len(train_y_raw)), train_y_raw] = 1
    
    # test_y = np.zeros((len(test_y_raw), 10))
    # test_y[np.arange(len(test_y_raw)), test_y_raw] = 1

    return (train_X, train_y), (test_X, test_y)


(train_X, train_y), (test_X, test_y) = mnist_dataset()

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])





model.fit(train_X, train_y, epochs=5)

pred_y = model.predict(test_X)

pred_ys_max_index = np.argmax(pred_y, axis=1)
test_ys_max_index = np.argmax(test_y, axis=1)

acc = list(map(lambda x: x[0] == x[1], zip(pred_ys_max_index, test_ys_max_index)))
acc = np.count_nonzero(acc)/float(len(acc))

print(acc)



# print(confusion_matrix(test_ys_max_index, pred_ys_max_index))
# print(acc_over_time)















# # model.evaluate(x_test, y_test)
