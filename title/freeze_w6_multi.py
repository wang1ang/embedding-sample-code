import tensorflow as tf
import argparse
import os
from cdssm import cdssm_tower
from content import content_net
from pprint import pprint
import config
from data import SentencePieceEmbedding

#checkpoint = r'E:\embedding\title\models\w6.multi\model.ckpt-865248'
#checkpoint = r'E:\embedding\title\models\w6.multi\model.ckpt-1118744'
# checkpoint = r'E:\embedding\title\models\w6.adam\model.ckpt-1041875'
# output_graph = r'E:\embedding\title\title_model_w6_adam.pb'
#checkpoint = r'E:\embedding\title\models\w6.l2\model.ckpt-1041136'
#output_graph = r'E:\embedding\title\title_model_w6_l2_longer.pb'
checkpoint = r'E:\embedding\title\models\w6.l2_multi\model.ckpt-850170'
output_graph = r'E:\embedding\title\title_model_w6_l2_multi.pb'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

conf = config.w6_7
def model(params):
    print ('vocab_size:', params['vocab_size'])
    mode = tf.estimator.ModeKeys.EVAL
    content = tf.placeholder(tf.float32, [None, 2048], 'content')
    doc_indices = tf.placeholder(tf.int64, [None, 2], 'doc_indices')
    doc_values = tf.placeholder(tf.int32, [None], 'doc_values')
    doc_length = tf.placeholder(tf.int32, [None], 'doc_length')


    batch_size = params['batch_size']
    doc_max_length = params['doc_max_length']
    embed_dim = params['embed_dim']
    conf = params['config']

    doc = tf.SparseTensor(indices=doc_indices, values=doc_values, dense_shape=(batch_size * doc_max_length, 64))
    doc_net = cdssm_tower(mode, doc, doc_length, doc_max_length, embed_dim, 'doc', params, conf['activation'])
    con_net = content_net(content, doc_net.shape[-1], mode, conf['activation'])
    
    # l2-normalize
    if conf['query_l2']:
        doc_net = tf.nn.l2_normalize(doc_net, axis=1, epsilon=1e-3, name='l2_normalize_query')
    if conf['con_l2']:
        con_net = tf.nn.l2_normalize(con_net, axis=1, epsilon=1e-3, name='l2_normalize_content')
    doc_net = tf.identity(doc_net, 'doc/output')
    con_net = tf.identity(con_net, 'content/output')
    similarities = conf['sfunc'](doc_net, con_net)
    similarities = tf.linalg.tensor_diag_part(similarities, name='similarity')
    return doc_net, con_net, similarities

def get_params():
    return {
        'vocab_size': SentencePieceEmbedding().size(),
        'batch_size': 1,
        'doc_max_length': 64, # doc max length
        'embed_dim': 128,
        'rnn_size': 512,
        'attention_size': 32,
        'hidden_units': [128, 32],
        'config': conf,
    }

config = tf.ConfigProto()
# occupy gpu gracefully
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    # load graph
    
    # option 1: from meta file
    #saver = tf.train.import_meta_graph(checkpoint+'.meta')
    
    # option 2: from predefined net
    doc_net, con_net, cosine = model(get_params())
    saver = tf.train.Saver()

    # load values
    saver.restore(sess, tf.train.latest_checkpoint(os.path.dirname(checkpoint)))
    
    # nodes
    graph = tf.get_default_graph()

    input_graph_def = graph.as_graph_def()
    output_graph_def = tf.graph_util.convert_variables_to_constants(
        sess,
        input_graph_def,
        ['doc/output', 'content/output', 'similarity']
    )
with tf.gfile.GFile(output_graph, "wb") as f:
    f.write(output_graph_def.SerializeToString())
print ('done')