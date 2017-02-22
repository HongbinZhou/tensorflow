
#Freeze model
def freeze_model():
    from tensorflow.python.tools import freeze_graph
    input_graph_path = "models/graph.pb"
    input_saver_def_path = ""
    input_binary = False
    checkpoint_path = "models/model.ckpt"
    output_node_names = "softmax"

    restore_op_name = "save/restore_all"  # set to default value, see: /usr/lib/python3.5/site-packages/tensorflow/python/tools/freeze_graph.py
    filename_tensor_name = "save/Const:0"  # set to default value

    output_graph_path = "models/frozen_graph.pb"
    clear_devices = False
    variable_names_blacklist = ""
    freeze_graph.freeze_graph(input_graph_path, input_saver_def_path,
                              input_binary, checkpoint_path, output_node_names,
                              restore_op_name, filename_tensor_name,
                              output_graph_path, clear_devices, variable_names_blacklist)

freeze_model()
