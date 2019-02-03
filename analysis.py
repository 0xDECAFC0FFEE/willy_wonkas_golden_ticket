import matplotlib.pyplot as plt
from pathlib import Path
from functools import reduce

def graph_accuracy(network_left_accs_list, name, filepath):
    plt.rcParams["figure.figsize"] = [16, 11]
    fig, ax = plt.subplots()
    for expr_num, (expr_train_accs, network_left) in enumerate(network_left_accs_list):
        color = expr_num/float(len(network_left_accs_list))
        line = ax.plot(expr_train_accs, label="%s%% - acc: %s" % (round(network_left, 2), round(expr_train_accs[-1], 4)), color=(0, color, 0), linewidth=.2)
    ax.legend()
    plt.title(name)
    plt.xlabel("epoch")
    plt.ylabel("acc")
    Path(*filepath).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(Path(*filepath))
    print("saved %s at %s" % (name, Path(*filepath)))

def draw_graph(layer_definitions, blacklists):
    for layer_definition, blacklist in zip(layer_definitions, blacklists):
        layer_type, params, init = layer_definition
        if layer_type == LayerType.input_1d:
            num_inputs = params["num_inputs"]
            prev_layer_size = num_inputs
            variables = []
        elif layer_type == LayerType.input_2d:
            image_width, image_height = params["image_width"], params["image_height"]
            prev_layer_size = (image_width, image_height)
            variables = []
        elif layer_type == LayerType.flatten:
            prev_layer_size = reduce(lambda a, b: a*b, prev_layer_size, 1)
            variables = []
        elif layer_type == LayerType.dropout:
            rate = params["rate"]
            prev_layer_output = tf.layers.dropout(prev_layer_output, rate)
            variables = []
        elif layer_type == LayerType.relu:
            layer_size = params["layer_size"]
            prev_layer_output, variables = fc_layer(prev_layer_output, tf.nn.relu, init, blacklist)
            prev_layer_size = layer_size
        elif layer_type == LayerType.softmax:
            layer_size = params["layer_size"]
            prev_layer_output, variables = fc_layer(prev_layer_output, tf.nn.softmax, init, blacklist)
            prev_layer_size = layer_size
        else:
            raise Exception("didnt find layer type", layer_type)
