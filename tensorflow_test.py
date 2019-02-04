import numpy as np
import tensorflow as tf
import random
from tqdm import tqdm
import os
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
import copy
import matplotlib.pyplot as plt
from collections import namedtuple
import random
import string
from pathlib import Path
from datetime import datetime
from sys import argv

from deep_model import LayerDefinition, set_layer_definitions_init_values, build_deep_model, build_blacklists, LayerType, get_layer_weights, count_nonzero_weights
from analysis import graph_accuracy, draw_graph
from io_funcs import mnist_dataset, load_NN, save_NN

# configuration
prune_percent = .2
num_epochs = 60 # 60
num_pruning_iterations = 40 # 20
# NN definition
init_layer_definitions = [
    LayerDefinition(type=LayerType.input_2d, params={"image_width":28, "image_height":28}),
    LayerDefinition(type=LayerType.flatten),
    LayerDefinition(type=LayerType.relu, params={"layer_size": 512}),
    LayerDefinition(type=LayerType.dropout, params={"rate": .2}),
    LayerDefinition(type=LayerType.softmax, params={"layer_size": 10})
]
num_weights_initial = set_layer_definitions_init_values(init_layer_definitions)
num_weights_left = num_weights_initial
blacklists = build_blacklists(init_layer_definitions)

# load saved network
saved_location = ["expr_records", "blacklist_modify_graph", 'exp_2019-02-03 04:38:12.695401_(60|30)_"more network weights = more stuff to prune? = higher final acc?"', "NN_definition"]
defaults = {}
blacklists, init_layer_definitions, num_weights_initial, num_weights_left = load_NN(saved_location, defaults)

start_timestamp = str(datetime.now())
expr_name = 'exp_%s-(%s|%s)-"%s"' % (start_timestamp, num_epochs, num_pruning_iterations, argv[1])
expr_record_path = ["expr_records", "blacklist_modify_graph", expr_name]
print('starting %s' % expr_name)

def batchify(Xs, ys, batches):
    Xs, ys = shuffle(Xs, ys, random_state=0)
    batch_size = int(len(Xs)/batches)
    for i in range(batches):
        yield Xs[batch_size*i: batch_size*(i+1)], ys[batch_size*i: batch_size*(i+1)]

def accuracy_1_hot(pred, true):
    pred_ys_max_index = np.argmax(pred, axis=1)
    true_ys_max_index = np.argmax(true, axis=1)
    acc = list(map(lambda x: x[0] == x[1], zip(pred_ys_max_index, true_ys_max_index)))
    acc = np.count_nonzero(acc)/float(len(acc))
    return acc

def experiment(num_epochs, layer_definitions, blacklist):
    (train_X, train_y), (val_X, val_y) = mnist_dataset()

    # build model from layer definition
    x, y, y_true, dnn_variables = build_deep_model(layer_definitions, blacklist)

    loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(y_true, y))
    train_step = tf.train.AdamOptimizer(learning_rate=.012).minimize(loss)
    init = tf.global_variables_initializer()

    # tpu_address = 'grpc://' + os.environ['COLAB_TPU_ADDR']
    # with tf.Session(tpu_address) as sess:
    with tf.Session() as sess:
        sess.run(init)

        batches = 8
        batch_size = int(len(train_X)/batches)

        expr_train_accs, expr_val_accs = [], []

        pbar = tqdm(range(num_epochs), leave=False, position=0)
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

        layer_weights = get_layer_weights(sess, dnn_variables)
        num_nonzero_weights = count_nonzero_weights(sess, dnn_variables)
        tqdm.write("%s nonzero layer weights after fitting" % num_nonzero_weights)
    tf.reset_default_graph()
    return layer_weights, expr_train_accs, expr_val_accs

# train and test an NN and update the blacklist
expr_train_accs_list, expr_val_accs_list = [], []
for run_num in tqdm(range(num_pruning_iterations), leave=False, position=1):
    layer_definitions = copy.deepcopy(init_layer_definitions)

    # train and test blacklisted NN
    weight_indices, expr_train_accs, expr_val_accs = experiment(num_epochs, layer_definitions, blacklists)
    expr_val_accs_list.append((expr_val_accs, num_weights_left/float(num_weights_initial) * 100))
    expr_train_accs_list.append((expr_train_accs, num_weights_left/float(num_weights_initial) * 100))

    # print final nn accuracy scores to terminal
    percent_weights_left = round(num_weights_left/float(num_weights_initial) *100, 2)
    tqdm.write("experiment %s val acc %s train acc %s %s%% (%s) network weights left" % (run_num, expr_val_accs[-1], expr_train_accs[-1], percent_weights_left, num_weights_left))

    # update blacklist with run results
    if run_num == num_pruning_iterations-1:
        break # don't update blacklist if its the last iteration for logging purposes
    weight_indices.sort(key=lambda x: abs(x[0]))
    num_weights_left_target = int(float(num_weights_left)*(1-prune_percent))
    for weight_index in weight_indices:
        weight, layer, wb, index = weight_index
        if blacklists[layer][wb][index]:
            blacklists[layer][wb][index] = False
            num_weights_left -= 1
            if num_weights_left_target >= num_weights_left:
                break

filename = expr_record_path + ["NN_definition"]
save_NN(filename, blacklists, init_layer_definitions, num_weights_initial, num_weights_left)
filename = expr_record_path + ["train.png"]
graph_accuracy(expr_train_accs_list, "blacklist with modified graph training accuracy", filename)
filename = expr_record_path + ["val.png"]
graph_accuracy(expr_val_accs_list, "blacklist with modified graph validation accuracy", filename)
# draw_graph(init_layer_definitions, blacklists)
