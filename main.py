from io_funcs import cifar10_dataset, batchify, mnist_dataset, get_exp_path
import numpy as np
import tensorflow as tf
print(tf.__version__)

tf.logging.set_verbosity(tf.logging.ERROR)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.allocator_type = 'BFC'

import random
from tqdm import tqdm
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from collections import namedtuple
import random
import string
from pathlib import Path

from deep_model import TF_Wrapper
from layer_definition import Input, Flatten, Dropout, ReLu, Softmax, Conv, Pool, Linear
from analysis import graph_accuracy, accuracy_1_hot#, draw_graph

# NN definition. almost all experiment parameters handled here
init_model = TF_Wrapper.new(
    layer_definitions=[
        Input(shape=[32, 32, 3]),
        # Conv(kernel_width=3, num_kernels=16),
        # Conv(kernel_width=3, num_kernels=16),
        Pool(),
        # Conv(kernel_width=3, num_kernels=32),
        # Conv(kernel_width=3, num_kernels=64),
        # Pool(),
        # Conv(kernel_width=3, num_kernels=128),
        # Conv(kernel_width=3, num_kernels=128),
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
        Flatten(),
        ReLu(layer_size=4098, prune_p=.2),
        Linear(layer_size=1024, prune_p=.2),
        ReLu(layer_size=4098, prune_p=.2),
        Softmax(layer_size=10, prune_p=.1)
    ],
    epochs=30,
    prune_iters=14,
    prune_style="local",
    global_prune_rate=.2,
    learning_rate=.0001
)

# train and test a network/blacklist/initial weight
def experiment(sess, model, dataset):
    (train_X, train_y), (val_X, val_y) = dataset

    x, y, y_true = model.x, model.y, model.y_true

    loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(y_true, y))
    train_step = tf.train.AdamOptimizer(learning_rate=model.learning_rate).minimize(loss)

    init = tf.global_variables_initializer()
    sess.run(init)

    batch_size = 100

    expr_train_accs, expr_val_accs = [], []

    pbar = tqdm(range(model.epochs), leave=False, position=1)
    for epoch in pbar:
        for Xs, ys in tqdm(list(batchify(train_X, train_y, batch_size)), leave=False, position=2):
            # train
            sess.run(train_step, feed_dict={ x: Xs, y_true: ys})

        # record epoch training acc. batching the train and test x values to fit in gpu
        pred_train_y = []
        for Xs, ys in batchify(train_X, train_y, batch_size, shuffle=False):
            pred_train_y.extend(sess.run(y, feed_dict={x: Xs}))
        train_acc = accuracy_1_hot(pred_train_y, train_y)
        expr_train_accs.append(train_acc)

        # record epoch testing acc. batching the train and test x values to fit in gpu
        pred_val_y = []
        for Xs, ys in batchify(val_X, val_y, batch_size, shuffle=False):
            pred_val_y.extend(sess.run(y, feed_dict={x: Xs}))
        val_acc = accuracy_1_hot(pred_val_y, val_y)
        expr_val_accs.append(val_acc)

        tqdm.write("train acc: %s val acc: %s" % (train_acc, val_acc))
        pbar.set_postfix({"val acc": "%s" % round(val_acc, 4)})

    num_nonzero_weights = model.num_weights(include_blacklist=False)
    tqdm.write("%s nonzero layer weights after fitting" % num_nonzero_weights)
    return expr_train_accs, expr_val_accs

# train and test an NN and update the blacklist
model = init_model
expr_train_accs_list, expr_val_accs_list = [], []
for run_num in tqdm(range(model.prune_iters), leave=False, position=0):
    with tf.Session(config=config) as sess:
        model = model.reinit()

        # train and test blacklisted NN
        expr_train_accs, expr_val_accs = experiment(sess, model, cifar10_dataset())
        num_weights_left = model.num_weights(include_blacklist=False)
        num_weights_initial = model.num_weights(include_blacklist=True)

        expr_val_accs_list.append((expr_val_accs, num_weights_left/float(num_weights_initial) * 100))
        expr_train_accs_list.append((expr_train_accs, num_weights_left/float(num_weights_initial) * 100))

        # tqdm.write final accuracy scores to terminal
        percent_weights_left = int(10000.0 * num_weights_left/float(init_model.num_weights(include_blacklist=False)))/100.0
        tqdm.write("experiment %s train acc %s val acc %s %s%% (%s) network weights left" % (run_num, expr_val_accs[-1], expr_train_accs[-1], percent_weights_left, num_weights_left))

        # update blacklist with trained weights.
        model.prune(sess)

    tf.reset_default_graph()

exp_path = get_exp_path()
filepath = exp_path + ["train.png"]
print(Path(*filepath))
graph_accuracy(expr_train_accs_list, "blacklist with modified graph training accuracy", filepath)
filepath = exp_path + ["val.png"]
graph_accuracy(expr_val_accs_list, "blacklist with modified graph validation accuracy", filepath)
filepath = exp_path + ["model"]
model.save(filepath)
# draw_graph(init_layer_definitions, blacklists)
