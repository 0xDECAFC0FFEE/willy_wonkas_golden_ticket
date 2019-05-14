import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from functools import reduce
# import graphviz

def accuracy_1_hot(pred, true):
    pred_ys_max_index = np.argmax(pred, axis=1)
    true_ys_max_index = np.argmax(true, axis=1)
    acc = list(map(lambda x: x[0] == x[1], zip(
        pred_ys_max_index, true_ys_max_index)))
    acc = np.count_nonzero(acc)/float(len(acc))
    return acc

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

# def draw_graph(init_layer_definitions, blacklists):
#     graph = graphviz.Graph(format="svg")
#     graph.attr("node", shape="box")
#     for layer_num, (layer_definition, blacklist) in enumerate(zip(init_layer_definitions, blacklists)):
#         layer_type, params, init = layer_definition
#         if layer_type == LayerType.input_1d:
#             num_inputs = params["num_inputs"]
#             prev_layer_size = num_inputs
#             variables = []
#         elif layer_type == LayerType.input_2d:
#             image_width, image_height = params["image_width"], params["image_height"]
#             prev_layer_size = (image_width, image_height)
#             variables = []
#         elif layer_type == LayerType.input_3d:
#             image_width, image_height = params["image_width"], params["image_height"]
#             image_depth = params["image_depth"]
#             prev_layer_size = (image_width, image_height, image_depth)
#             variables = []
#         elif layer_type == LayerType.flatten:
#             prev_layer_size = reduce(lambda a, b: a*b, prev_layer_size, 1)
#             variables = []
#         elif layer_type == LayerType.dropout:
#             variables = []
#             for edge_num in range(prev_layer_size):
#                 graph.edge(str((layer_num-1, edge_num)), str((layer_num, edge_num)), len="90")

#         elif layer_type in [LayerType.relu, LayerType.softmax]:
#             layer_size = params["layer_size"]
#             for node_num, node in enumerate(blacklist[0]):
#                 for edge_num, edge in enumerate(node):
#                     if edge:
#                         graph.edge(str((layer_num-1, node_num)), str((layer_num, edge_num)), len="90")
#             prev_layer_size = layer_size
#         else:
#             raise Exception("didnt find layer type", layer_type)
#     graph.render("model_graph", view=True)
#     print("saved graph image at model_graph.png")
