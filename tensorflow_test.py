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

from deep_model import LayerParam, set_layer_param_init_values, build_deep_model, build_blacklist, update_blacklist

def batchify(Xs, ys, batches):
    Xs, ys = shuffle(Xs, ys, random_state=0)
    batch_size = int(len(Xs)/batches)
    for i in range(batches):
        yield Xs[batch_size*i: batch_size*(i+1)], ys[batch_size*i: batch_size*(i+1)]

def mnist_dataset():
    raw_data = tf.keras.datasets.mnist.load_data(path='mnist.npz')
    train_X, train_y_raw = raw_data[0]
    val_X, val_y_raw = raw_data[1]

    train_X = train_X.reshape((len(train_X), 784))
    val_X = val_X.reshape((len(val_X), 784))

    # normalizing x values in [0, 1]
    train_X = train_X/255.0
    val_X = val_X/255.0

    # 1 hot encoding y values
    train_y = np.zeros((len(train_y_raw), 10))
    train_y[np.arange(len(train_y_raw)), train_y_raw] = 1
    val_y = np.zeros((len(val_y_raw), 10))
    val_y[np.arange(len(val_y_raw)), val_y_raw] = 1

    return (train_X, train_y), (val_X, val_y)

WeightIndex = namedtuple("WeightIndex", ["weight", "layer", "wb", "index"])

def get_layer_weights(sess, dnn_variables):
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

def experiment(layers_params, blacklisted):
    (train_X, train_y), (val_X, val_y) = mnist_dataset()

    x, y, y_true, dnn_variables = build_deep_model(layers_params)

    loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(y_true, y))
    train_step = tf.train.AdamOptimizer(learning_rate=.01).minimize(loss)
    init = tf.global_variables_initializer()

    # tpu_address = 'grpc://' + os.environ['COLAB_TPU_ADDR']
    # with tf.Session(tpu_address) as sess:
    with tf.Session() as sess:
        acc_over_time = []

        sess.run(init)
        layer_weights = get_layer_weights(sess, dnn_variables)

        batches = 8
        batch_size = int(len(train_X)/batches)

        for epoch in range(200): #200
            for Xs, ys in batchify(train_X, train_y, batches):
                # Train
                sess.run(train_step, feed_dict={ x: Xs, y_true: ys})
                for variable_layer, blacklisted_layer in zip(dnn_variables, blacklisted):
                    for variable, blacklisted_wb in zip(variable_layer, blacklisted_layer):
                        variable.load(np.where(blacklisted_wb, sess.run(variable), 0), sess)

        # check test acc
            pred_ys_max_index = np.argmax(sess.run(y, feed_dict={x: val_X}), axis=1)
            val_ys_max_index = np.argmax(val_y, axis=1)

            val_acc = list(map(lambda x: x[0] == x[1], zip(pred_ys_max_index, val_ys_max_index)))
            val_acc = np.count_nonzero(val_acc)/float(len(val_acc))
            acc_over_time.append(val_acc)

        layer_weights = get_layer_weights(sess, dnn_variables)
        return acc_over_time, layer_weights

init_layers_params = [
    LayerParam(type="input", params=784),
    LayerParam(type="dense", params=512), 
    LayerParam(type="dropout", params=.2), 
    LayerParam(type="output", params=10)
]
set_layer_param_init_values(init_layers_params)
blacklisted, num_weights = build_blacklist(init_layers_params)
num_weights_initial = num_weights

prune_percent = .2
accs_over_time = []

for experiment_num in tqdm(range(3)):
    layers_params = copy.deepcopy(init_layers_params)
    for layer_params, layer_blacklist in zip(layers_params, blacklisted):
        if layer_params.init == None:
            continue
        for layer_param, layer_blacklisted_wb in zip(layer_params.init, layer_blacklist):
            layer_param = np.where(layer_blacklisted_wb, layer_param, 0)

    acc_over_time, weight_indices = experiment(layers_params, blacklisted)
    accs_over_time.append((acc_over_time, num_weights/num_weights_initial * 100))
    print("experiment %s final acc %s %s%% network weights left" % (experiment_num, acc_over_time[-1], num_weights/num_weights_initial *100))
    weight_indices.sort(key=lambda x: abs(x[0]))

    prune_num = (num_weights)*prune_percent
    for weight_index in weight_indices:
        weight, layer, wb, index = weight_index

        if blacklisted[layer][wb][index] == True:
            blacklisted[layer][wb][index] = False
            num_weights -= 1
            prune_num -= 1
            if prune_num < 1:
                break

lines = []
fig, ax = plt.subplots()
for expr_num, (acc_over_time, network_left) in enumerate(accs_over_time):
    color = expr_num/float(len(accs_over_time))
    line = ax.plot(acc_over_time, label="%s%%" % network_left, color=(0, color, 0), linewidth=.2)

ax.legend()
plt.title("blacklist with weights updated each epoch")
plt.show()