import tensorflow as tf
import argparse
import os
from cdssm import bert
from content import content_net
from pprint import pprint
import config
from data import SentencePieceEmbedding

checkpoint = r'E:\embedding\title\models\b1_multi\model.ckpt-818611'
output_graph = r'E:\embedding\title\title_model_b1_multi_longer.pb'
bert_config = r'bert\xlm_bert_convert_dis_query_layer3\xlm_config_dis.json'
content_only = False
os.environ['CUDA_VISIBLE_DEVICES'] = ''

conf = config.b1_1
def model(params):
    print ('vocab_size:', params['vocab_size'])
    mode = tf.estimator.ModeKeys.EVAL
    content = tf.placeholder(tf.float32, [None, 2048], 'content')

    batch_size = params['batch_size']
    doc_max_length = params['doc_max_length']
    embed_dim = params['embed_dim']
    conf = params['config']
    print ('doc_max_length', doc_max_length)
    input_ids = tf.placeholder(tf.int64, [None, doc_max_length], 'doc_ids')
    input_mask = tf.placeholder(tf.int32, [None, doc_max_length], 'doc_mask')
    input_type = tf.placeholder(tf.int32, [None, doc_max_length], 'doc_type')

    doc_net = bert(bert_config, mode, params['hidden_units'][-1], input_ids, input_mask, input_type, conf['activation'])
    con_net = content_net(content, doc_net.shape[-1], mode, conf['activation'])
    
    # use un-normalized this time
    doc_net = tf.identity(doc_net, 'doc/output')
    con_net = tf.identity(con_net, 'content/output')
    # l2-normalize
    if conf['query_l2']:
        doc_net = tf.nn.l2_normalize(doc_net, axis=1, epsilon=1e-3, name='l2_normalize_query')
    if conf['con_l2']:
        con_net = tf.nn.l2_normalize(con_net, axis=1, epsilon=1e-3, name='l2_normalize_content')
    similarities = conf['sfunc'](doc_net, con_net)
    similarities = tf.linalg.tensor_diag_part(similarities, name='similarity')
    return doc_net, con_net, similarities

def get_params():
    return {
        'vocab_size': SentencePieceEmbedding().size(),
        'batch_size': 1,
        'doc_max_length': 12, # doc max length
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
    if content_only:
        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess,
            input_graph_def,
            ['content/output']
        )
        output_graph = output_graph.replace('.pb', '_content.pb')
    else:
        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess,
            input_graph_def,
            ['doc/output', 'content/output', 'similarity']
        )
with tf.gfile.GFile(output_graph, "wb") as f:
    f.write(output_graph_def.SerializeToString())
print ('done')