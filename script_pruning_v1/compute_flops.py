import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.contrib.framework.python.framework import checkpoint_utils as cp
from model import Model_DIN_V2_Gru_Vec_attGru_Neg
def load_pb(pb):
    with tf.gfile.GFile(pb, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='')
        return graph

def list_variables(path):
    var_list = cp.list_variables(path)
    for v in var_list:
        print(v)


def freeze_graph(model_folder):
    checkpoint = tf.train.get_checkpoint_state(model_folder)
    input_checkpoint = checkpoint.model_checkpoint_path
    
    absolute_model_folder = "/".join(input_checkpoint.split('/')[:-1])
    output_graph = absolute_model_folder + "/frozen_model.pb"
    output_node_names = "output"
 
    clear_devices = True
    
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=clear_devices)
 
    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()
    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint)
 
        # We use a built-in TF helper to export variables to constant
        output_graph_def = graph_util.convert_variables_to_constants(
            sess, 
            input_graph_def, 
            output_node_names.split(",") # We split on comma for convenience
        ) 
 
        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph." % len(output_graph_def.node))
  


if __name__ == '__main__':
    #tf.reset_default_graph()
    #with tf.Session() as sess:
        #saver = tf.train.import_meta_graph('/home/rihan.crh/dien/dnn_best_model/ckpt_noshuffDIEN3.meta')
        #graph_def = tf.get_default_graph().as_graph_def()
        #node_list=[n.name for n in graph_def.node]
        #for each in node_list:
            #print(each)
    
    #freeze_graph("/home/rihan.crh/dien/dnn_best_model")
    g = load_pb("/home/rihan.crh/dien/dnn_best_model/frozen_model.pb")
    with g.as_default():
        flops = tf.profiler.profile(g, options = tf.profiler.ProfileOptionBuilder.float_operation())
        print('FLOP after freezing %d'%flops.total_float_ops)
            
