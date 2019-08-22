import tensorflow as tf
import argparse
import os
from cdssm import cdssm_tower
from content import content_net
from data import trilettergram
from pprint import pprint

checkpoint = r'E:\embedding\title\models\v1\model.ckpt-1057926'
output_graph = r'E:\embedding\title\title_model_v1.pb'
version = 'v1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

def model(params, version):
    mode = tf.estimator.ModeKeys.EVAL

    content = tf.placeholder(tf.float32, [None, 2048], 'content')
    doc_indices = tf.placeholder(tf.int64, [None, 2], 'doc_indices')
    doc_values = tf.placeholder(tf.int32, [None], 'doc_values')
    doc_length = tf.placeholder(tf.int32, [None], 'doc_length')

    doc = tf.SparseTensor(indices=doc_indices, values=doc_values, dense_shape=(params['batch_size'] * params['doc_max_length'], 64))
    doc_net = cdssm_tower(mode, doc, doc_length, params['doc_max_length'], params['embed_dim'], 'doc', params, version)
    doc_net = tf.identity(doc_net, 'doc/output')
    con_net = content_net(content, doc_net.shape[-1], mode, version)
    con_net = tf.identity(con_net, 'content/output')

    with tf.variable_scope('self_cosine_loss'):
        # suppose query & doc are l2-normalized
        cosines = tf.matmul(doc_net, con_net, transpose_b=True)
    cosines = tf.linalg.tensor_diag_part(cosines, name='cosine')
    return doc_net, con_net, cosines

def get_params():
    vocab = trilettergram()
    return {
        'vocab_size': len(vocab),
        'batch_size': 1,
        'doc_max_length': 64, # doc max length
        'embed_dim': 128,
        'rnn_size': 512,
        'attention_size': 32,
        'hidden_units': [128, 32],
    }

config = tf.ConfigProto()
# occupy gpu gracefully
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    # load graph
    
    # option 1: from meta file
    #saver = tf.train.import_meta_graph(checkpoint+'.meta')
    
    # option 2: from predefined net
    doc_net, con_net, cosine = model(get_params(), version)
    saver = tf.train.Saver()

    # load values
    saver.restore(sess, tf.train.latest_checkpoint(os.path.dirname(checkpoint)))
    
    # nodes
    graph = tf.get_default_graph()

    input_graph_def = graph.as_graph_def()
    output_graph_def = tf.graph_util.convert_variables_to_constants(
        sess,
        input_graph_def,
        #['doc/l2_normalize', 'fc/pred_dssm/Relu', 'cosine']
        ['doc/output', 'content/output', 'cosine']
    )
with tf.gfile.GFile(output_graph, "wb") as f:
    f.write(output_graph_def.SerializeToString())
print ('done')