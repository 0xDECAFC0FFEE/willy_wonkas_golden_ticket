import tensorflow as tf
import numpy as np
from datetime import datetime
from pathlib import Path
import pickle
from sklearn import utils
from sys import argv
import datetime

def get_exp_path():
    now = datetime.datetime.now()
    try:
        exp_name = argv[1]
    except:
        exp_name = "test"

    exp_name = 'experiment %s-%s-%s %s:%s:%s "%s"' % (now.month, now.day, now.year, now.hour, now.minute, now.second, exp_name)
    exp_path = ["expr_records", exp_name]

    return exp_path

def batchify(Xs, ys, batch_size, shuffle=True):
    if shuffle:
        Xs, ys = utils.shuffle(Xs, ys, random_state=0)

    batches = int(len(Xs)/batch_size)
    for i in range(batches):
        yield Xs[batch_size*i: batch_size*(i+1)], ys[batch_size*i: batch_size*(i+1)]

def mnist_dataset():
    raw_data = tf.keras.datasets.mnist.load_data()
    train_X, train_y_raw = raw_data[0]
    val_X, val_y_raw = raw_data[1]

    # normalizing x values in [0, 1]
    train_X = train_X/255.0
    val_X = val_X/255.0

    train_y_raw = train_y_raw.flatten()
    val_y_raw = val_y_raw.flatten()

    # 1 hot encoding y values
    train_y = np.zeros((len(train_y_raw), 10))
    train_y[np.arange(len(train_y_raw)), train_y_raw] = 1
    val_y = np.zeros((len(val_y_raw), 10))
    val_y[np.arange(len(val_y_raw)), val_y_raw] = 1

    return (train_X, train_y), (val_X, val_y)


def cifar10_dataset():
    raw_data = tf.keras.datasets.cifar10.load_data()
    train_X, train_y_raw = raw_data[0]
    val_X, val_y_raw = raw_data[1]

    # normalizing x values in [0, 1]
    train_X = train_X/255.0
    val_X = val_X/255.0

    train_y_raw = train_y_raw.flatten()
    val_y_raw = val_y_raw.flatten()

    # 1 hot encoding y values
    train_y = np.zeros((len(train_y_raw), 10))
    train_y[np.arange(len(train_y_raw)), train_y_raw] = 1
    val_y = np.zeros((len(val_y_raw), 10))
    val_y[np.arange(len(val_y_raw)), val_y_raw] = 1

    return (train_X, train_y), (val_X, val_y)

# def load_NN(path, defaults):
#     a = "init_layer_definitions" in defaults
#     b = "blacklists" in defaults
#     assert((not a and not b) or (a and b))

#     with open(Path(*path), "rb") as file:
#         saved_values = pickle.load(file)
#         a = "init_layer_definitions" in saved_values
#         b = "blacklists" in saved_values
#         assert((not a and not b) or (a and b))
#     defaults.update(saved_values)

#     blacklists = defaults["blacklists"]
#     init_layer_definitions = defaults["init_layer_definitions"]
#     num_weights = defaults["num_weights"]
#     num_weights_left = defaults["num_weights_left"]
#     return blacklists, init_layer_definitions, num_weights, num_weights_left

# def save_NN(path, blacklists, init_layer_definitions, num_weights, num_weights_left):
#     Path(*path).parent.mkdir(parents=True, exist_ok=True)
#     with open(Path(*path), "w+b") as file_hanlde:
#         saved_values = {
#             "blacklists": blacklists,
#             "init_layer_definitions": init_layer_definitions,
#             "num_weights": num_weights,
#             "num_weights_left": num_weights_left
#         }
#         pickle.dump(saved_values, file)
#     tqdm.write("saved nn at %s" % file)

