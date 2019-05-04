from io_funcs import cifar10_dataset, load_NN, save_NN, batchify, mnist_dataset
import numpy as np
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

import random
from tqdm import tqdm
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from sklearn.metrics import confusion_matrix
import copy
import matplotlib.pyplot as plt
from collections import namedtuple
import random
import string
from pathlib import Path
import datetime
from sys import argv

from deep_model import TF_Wrapper
from layer_definition import Input, Flatten, Dropout, ReLu, Softmax, Conv, Pool
from analysis import graph_accuracy, accuracy_1_hot#, draw_graph

# NN definition
init_model = TF_Wrapper.new(
    layer_definitions=[
        Input(shape=[32, 32, 3]),
        Conv(kernel_width=3, num_kernels=64),
        Conv(kernel_width=3, num_kernels=64),
        Pool(),
        Conv(kernel_width=3, num_kernels=128),
        Conv(kernel_width=3, num_kernels=128),
        # Pool(),
        # Conv(kernel_width=3, num_kernels=256),
        # Conv(kernel_width=3, num_kernels=256),
        # Conv(kernel_width=3, num_kernels=256),
        # Pool(),
        # Conv(kernel_width=3, num_kernels=512),
        # Conv(kernel_width=3, num_kernels=512),
        # Conv(kernel_width=3, num_kernels=512),
        # Pool(),
        # Conv(kernel_width=3, num_kernels=512),
        # Conv(kernel_width=3, num_kernels=512),
        Pool(),
        Flatten(),
        ReLu(layer_size=300, prune_p=.2),
        ReLu(layer_size=300, prune_p=.2),
        ReLu(layer_size=100, prune_p=.2),
        Softmax(layer_size=10, prune_p=.1)
    ],
    epochs=30,
    prune_iters=20,
    prune_style="global",
    global_prune_rate=.2
)

now = datetime.datetime.now()
try:
    expr_name = argv[1]
except:
    expr_name = "test"
expr_name = 'experiment %s/%s/%s %s:%s "%s"' % (now.month, now.day, now.year, now.minute, now.second, expr_name)
expr_record_path = ["expr_records", expr_name]
print('starting %s' % expr_name)

def experiment(sess, model, dataset):
    (train_X, train_y), (val_X, val_y) = dataset

    print("train y shape", train_y.shape)

    # build model from layer definition
    x, y, y_true = model.x, model.y, model.y_true

    loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(y_true, y))
    train_step = tf.train.AdamOptimizer(learning_rate=.0007).minimize(loss)
    init = tf.global_variables_initializer()

    sess.run(init)

    batches = 8
    batch_size = int(len(train_X)/batches)

    expr_train_accs, expr_val_accs = [], []

    pbar = tqdm(range(model.epochs), leave=False, position=0)
    for epoch in pbar:
        for Xs, ys in batchify(train_X, train_y, batches):
            # Train
            sess.run(train_step, feed_dict={ x: Xs, y_true: ys})

        # accuracy measures
        train_acc = accuracy_1_hot(sess.run(y, feed_dict={x: train_X}), train_y)
        expr_train_accs.append(train_acc)
        val_acc = accuracy_1_hot(sess.run(y, feed_dict={x: val_X}), val_y)
        expr_val_accs.append(val_acc)
        pbar.set_postfix({"val acc": "%s" % round(val_acc, 4)})

    num_nonzero_weights = model.num_weights(include_blacklist=False)
    tqdm.write("%s nonzero layer weights after fitting" % num_nonzero_weights)
    return expr_train_accs, expr_val_accs

# train and test an NN and update the blacklist
model = init_model
expr_train_accs_list, expr_val_accs_list = [], []
for run_num in tqdm(range(model.prune_iters), leave=False, position=1):
    with tf.Session() as sess:
        model = model.copy()

        # train and test blacklisted NN
        expr_train_accs, expr_val_accs = experiment(sess, model, cifar10_dataset())
        num_weights_left = model.num_weights(include_blacklist=False)
        num_weights_initial = model.num_weights(include_blacklist=True)

        expr_val_accs_list.append((expr_val_accs, num_weights_left/float(num_weights_initial) * 100))
        expr_train_accs_list.append((expr_train_accs, num_weights_left/float(num_weights_initial) * 100))

        # print final nn accuracy scores to terminal
        percent_weights_left = 100.0 * num_weights_left/float(init_model.num_weights(include_blacklist=False))
        tqdm.write("experiment %s val acc %s train acc %s %s%% (%s) network weights left" % (run_num, expr_val_accs[-1], expr_train_accs[-1], percent_weights_left, num_weights_left))

        # update blacklist with trained weights
        model.prune(sess)

    tf.reset_default_graph()

# filename = expr_record_path + ["NN_definition"]
# save_NN(filename, blacklists, init_layer_definitions, num_weights_initial, num_weights_left)
filename = expr_record_path + ["train.png"]
graph_accuracy(expr_train_accs_list, "blacklist with modified graph training accuracy", filename)
filename = expr_record_path + ["val.png"]
graph_accuracy(expr_val_accs_list, "blacklist with modified graph validation accuracy", filename)
# draw_graph(init_layer_definitions, blacklists)
