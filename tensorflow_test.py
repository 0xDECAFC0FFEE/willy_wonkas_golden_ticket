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
from collections import namedtuple

from deep_model import LayerDefinition, set_layer_definitions_init_values, build_deep_model, build_blacklists, LayerType

def batchify(Xs, ys, batches):
    Xs, ys = shuffle(Xs, ys, random_state=0)
    batch_size = int(len(Xs)/batches)
    for i in range(batches):
        yield Xs[batch_size*i: batch_size*(i+1)], ys[batch_size*i: batch_size*(i+1)]

def mnist_dataset():
    raw_data = tf.keras.datasets.mnist.load_data(path='mnist.npz')
    train_X, train_y_raw = raw_data[0]
    val_X, val_y_raw = raw_data[1]

    # normalizing x values in [0, 1]
    train_X = train_X/255.0
    val_X = val_X/255.0

    # 1 hot encoding y values
    train_y = np.zeros((len(train_y_raw), 10))
    train_y[np.arange(len(train_y_raw)), train_y_raw] = 1
    val_y = np.zeros((len(val_y_raw), 10))
    val_y[np.arange(len(val_y_raw)), val_y_raw] = 1

    return (train_X, train_y), (val_X, val_y)

def get_layer_weights(sess, dnn_variables):
    WeightIndex = namedtuple("WeightIndex", ["weight", "layer", "wb", "index"])

    def get_weights(i_l, i_wb, i_cur, weights):
        if len(weights.shape) > 1:
            weight_indices = []
            for i, weight in enumerate(weights):
                c_weights = get_weights(i_l, i_wb, i_cur+(i,), weight)
                weight_indices.extend(c_weights)
            return weight_indices
        elif len(weights.shape) == 1:
            return [WeightIndex(weight, i_l, i_wb, i_cur+(i,)) for i, weight in enumerate(weights)]
        else:
            raise Exception("unexpected weights value %s" % weighs)

    variables = []
    for i_l, layer in enumerate(dnn_variables):
        for i_wb, weight_variables in enumerate(layer):
            weight_values = sess.run(weight_variables)
            variables.extend(get_weights(i_l, i_wb, tuple(), weight_values))

    return variables

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
        return layer_weights, expr_train_accs, expr_val_accs

prune_percent = .2
num_epochs = 100 #50
num_pruining_iterations = 20 # 20

# define neural network
init_layer_definitions = [
    LayerDefinition(type=LayerType.input_2d, params={"image_width":28, "image_height":28}),
    LayerDefinition(type=LayerType.flatten),
    LayerDefinition(type=LayerType.relu, params={"layer_size": 512}),
    LayerDefinition(type=LayerType.dropout, params={"rate": .2}),
    LayerDefinition(type=LayerType.softmax, params={"layer_size": 10})
]
num_weights_initial = set_layer_definitions_init_values(init_layer_definitions)
blacklists = build_blacklists(init_layer_definitions)

# train and test an NN and update the blacklist
expr_train_accs_list, expr_val_accs_list = [], []
num_weights_left = num_weights_initial
for experiment_num in tqdm(range(num_pruining_iterations), leave=False, position=1):
    layer_definitions = copy.deepcopy(init_layer_definitions)

    # train and test blacklisted NN
    weight_indices, expr_train_accs, expr_val_accs = experiment(num_epochs, layer_definitions, blacklists)
    expr_val_accs_list.append((expr_val_accs, num_weights_left/float(num_weights_initial) * 100))
    expr_train_accs_list.append((expr_train_accs, num_weights_left/float(num_weights_initial) * 100))

    # output current NN scores
    round(num_weights_left/float(num_weights_initial) *100, 2)
    tqdm.write("experiment %s val acc %s train acc %s %s%% (%s) network weights left" % (experiment_num, expr_val_accs[-1], expr_train_accs[-1], percent_weights_left, num_weights_left))

    # update blacklist
    weight_indices.sort(key=lambda x: abs(x[0]))
    num_weights_left_target = int(float(num_weights_left)*(1-prune_percent))
    for weight_index in weight_indices:
        weight, layer, wb, index = weight_index
        if blacklists[layer][wb][index]:
            blacklists[layer][wb][index] = False
            num_weights_left -= 1
            if num_weights_left_target >= num_weights_left:
                break


# graph accuracy
fig, ax = plt.subplots()
for expr_num, (expr_train_accs, network_left) in enumerate(expr_train_accs_list):
    color = expr_num/float(len(expr_train_accs_list))
    line = ax.plot(expr_train_accs, label="%s%% - acc: %s" % (round(network_left, 2), round(expr_train_accs[-1], 4)), color=(0, color, 0), linewidth=.2)
ax.legend()
plt.title("blacklist with modified graph training accuracy")
plt.xlabel("epoch")
plt.ylabel("train acc")
plt.show()

fig, ax = plt.subplots()
for expr_num, (expr_val_accs, network_left) in enumerate(expr_val_accs_list):
    color = expr_num/float(len(expr_val_accs_list))
    line = ax.plot(expr_val_accs, label="%s%% - acc: %s" % (round(network_left, 2), round(expr_val_accs[-1], 4)), color=(0, color, 0), linewidth=.2)
ax.legend()
plt.title("blacklist with modified graph validation accuracy")
plt.xlabel("epoch")
plt.ylabel("val acc")
plt.show()