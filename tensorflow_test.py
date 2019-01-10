import numpy as np
import tensorflow as tf
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
import copy
import matplotlib.pyplot as plt

def batchify(Xs, ys, batches):
    Xs, ys = shuffle(Xs, ys, random_state=0)
    batch_size = int(len(Xs)/batches)
    for i in range(batches):
        yield Xs[batch_size*i: batch_size*(i+1)], ys[batch_size*i: batch_size*(i+1)]

def mnist_dataset():
    raw_data = tf.keras.datasets.mnist.load_data(path='mnist.npz')
    train_X, train_y_raw = raw_data[0]
    test_X, test_y_raw = raw_data[1]

    train_X = train_X.reshape((len(train_X), 784))
    test_X = test_X.reshape((len(test_X), 784))

    # normalizing x values in [0, 1]
    train_X = train_X/255.0
    test_X = test_X/255.0

    # 1 hot encoding y values
    train_y = np.zeros((len(train_y_raw), 10))
    train_y[np.arange(len(train_y_raw)), train_y_raw] = 1
    test_y = np.zeros((len(test_y_raw), 10))
    test_y[np.arange(len(test_y_raw)), test_y_raw] = 1

    return (train_X, train_y), (test_X, test_y)


def build_deep_model(num_inputs, layer_parameters, num_outputs):
    x = tf.placeholder(tf.float32, [None, num_inputs])
    y_true = tf.placeholder(tf.float32, [None, num_outputs])

    def sigmoid_layer(X, prev_layer_size, layer_size):
        with tf.name_scope('relu_node'):
            # random normal and 0 distributed variables
            # W = tf.Variable(tf.random_normal([prev_layer_size, layer_size]))
            # b = tf.Variable(tf.zeros([layer_size]))

            # xavier initialized variables
            W = tf.get_variable("W", shape=[prev_layer_size, layer_size], initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable("b", shape=[layer_size], initializer=tf.contrib.layers.xavier_initializer())
            return tf.nn.relu(tf.matmul(X, W) + b)

    def tanh_layer(X, prev_layer_size, num_outputs):
        with tf.name_scope('softmax_node'):
            # random normal and 0 distributed variables
            # W = tf.Variable(tf.random_normal([prev_layer_size, num_outputs]))
            # b = tf.Variable(tf.zeros([num_outputs]))

            # xavier initialized variables
            W = tf.get_variable("W2", shape=[prev_layer_size, num_outputs], initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable("b2", shape=[num_outputs], initializer=tf.contrib.layers.xavier_initializer())
            return tf.nn.softmax(tf.matmul(X, W) + b)

    prev_layer_size = num_inputs
    prev_layer_output = x
    for layer_desc in layer_parameters:
        layer_type, params = layer_desc
        if layer_type == "dropout":
            rate = params
            prev_layer_output = tf.layers.dropout(prev_layer_output, rate)
        elif layer_type == "dense":
            layer_size = params
            prev_layer_output = sigmoid_layer(prev_layer_output, prev_layer_size, layer_size)
            prev_layer_size = layer_size

    return x, tanh_layer(prev_layer_output, prev_layer_size, num_outputs), y_true




(train_X, train_y), (test_X, test_y) = mnist_dataset()

x, y, y_true = build_deep_model(784, [("dense", 512), ("dropout", .2)], 10)

# squared difference
# loss = tf.reduce_mean(tf.square(y_true-y))

# categorical crossentropy
loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(y_true, y))

tf.summary.histogram("loss", loss)

train_step = tf.train.AdamOptimizer(learning_rate=.01).minimize(loss)

init = tf.global_variables_initializer()

# tpu_address = 'grpc://' + os.environ['COLAB_TPU_ADDR']
# with tf.Session(tpu_address) as sess:
with tf.Session() as sess:
    acc_over_time = []
    # train_writer = tf.summary.FileWriter( './logs', sess.graph)

    sess.run(init)

    batches = 8
    batch_size = int(len(train_X)/batches)

    for epoch in tqdm(range(10)): #200
        for Xs, ys in batchify(train_X, train_y, batches):
            # Train
            sess.run(train_step, feed_dict={ x: Xs, y_true: ys})

            # merge = tf.summary.merge_all()
            # summary, _ = sess.run([merge, train_step], feed_dict={ x: Xs, y_true: ys})
            # train_writer.add_summary(summary, epoch)

    # check test acc
        pred_ys_max_index = np.argmax(sess.run(y, feed_dict={x: test_X}), axis=1)
        test_ys_max_index = np.argmax(test_y, axis=1)

        test_acc = list(map(lambda x: x[0] == x[1], zip(pred_ys_max_index, test_ys_max_index)))
        test_acc = np.count_nonzero(test_acc)/float(len(test_acc))
        print("test test_acc", test_acc)
        acc_over_time.append(test_acc)

    print(confusion_matrix(test_ys_max_index, pred_ys_max_index))
    plt.plot(acc_over_time)
    plt.show()
