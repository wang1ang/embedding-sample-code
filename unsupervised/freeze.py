import tensorflow as tf
import argparse
import os

from pprint import pprint

def hidden_layer(x, dim, scope):
    return tf.layers.dense(x, units=dim, activation=tf.nn.tanh, name=scope)
def output_layer(x, dim, scope):
    return tf.layers.dense(x, units=dim, activation=tf.nn.relu, name=scope)

def model_v7():
    # content
    feature = tf.placeholder(tf.float32, shape=(None, 2048), name='Content')
    layer1 = hidden_layer(feature, 4096, 'fc/content_fc1')
    layer2 = hidden_layer(layer1, 1024, 'fc/content_fc2')
    content = layer2
    pred_dan = output_layer(content, 25, scope='fc/fc_dan')
    pred_cdssm = output_layer(content, 32, scope='fc/fc_cdssm')
    return pred_dan, pred_cdssm

#checkpoint = r'log5\model.ckpt-2026339'
checkpoint = r'log7_traintest\model.ckpt-1040002'
output_graph = './model_v7.pb'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
parser = argparse.ArgumentParser(description='Tensorflow Pose Estimation Graph Extractor')
parser.add_argument('--checkpoint', type=str, default=checkpoint, help='checkpoint path')
parser.add_argument('--output_node_names', type=str, default='fc/fc_dan/BiasAdd,fc/fc_cdssm/BiasAdd') # changed
parser.add_argument('--output_graph', type=str, default=output_graph, help='output_freeze_path')

args = parser.parse_args()

config = tf.ConfigProto()
# occupy gpu gracefully
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    # load graph
    
    # option 1: from meta file
    #new_saver = tf.train.import_meta_graph(args.checkpoint+'.meta')
    
    # option 2: from predefined net
    dan, cdssm = model_v7()
    saver = tf.train.Saver()

    # load values
    saver.restore(sess, tf.train.latest_checkpoint(os.path.dirname(args.checkpoint)))
    
    # nodes
    graph = tf.get_default_graph()

    input_graph_def = graph.as_graph_def()
    output_graph_def = tf.graph_util.convert_variables_to_constants(
        sess,
        input_graph_def,
        args.output_node_names.split(",")
    )
with tf.gfile.GFile(args.output_graph, "wb") as f:
    f.write(output_graph_def.SerializeToString())