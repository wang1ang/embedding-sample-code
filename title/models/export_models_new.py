import tensorflow as tf
import os
import sys
sys.path.append('../')
from cdssm import cdssm_tower
from content import content_net
from pprint import pprint
import config
from data import SentencePieceEmbedding

# for export
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import utils

checkpoint = r'E:\embedding\title\models\w6.l2\fix\model.ckpt-852565'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

conf = config.w6_7
batch_size = 1
def model(doc_indices, doc_values, doc_length, params):
    print ('vocab_size:', params['vocab_size'])
    mode = tf.estimator.ModeKeys.EVAL

    #ids = tf.placeholder(tf.int32, [None], 'input_ids')
    #length = 

    batch_size = params['batch_size']
    doc_max_length = params['doc_max_length']
    embed_dim = params['embed_dim']
    conf = params['config']

    doc = tf.SparseTensor(indices=doc_indices, values=doc_values, dense_shape=(batch_size * doc_max_length, 64))
    doc_net = cdssm_tower(mode, doc, doc_length, doc_max_length, embed_dim, 'doc', params, conf['activation'])
    
    # l2-normalize
    if conf['query_l2']:
        doc_net = tf.nn.l2_normalize(doc_net, axis=1, epsilon=1e-3, name='l2_normalize_query')
    doc_net = tf.identity(doc_net, 'doc/output')
    return doc_net

def get_params():
    return {
        'vocab_size': SentencePieceEmbedding().size(),
        'batch_size': batch_size,
        'doc_max_length': 64, # doc max length
        'embed_dim': 128,
        'rnn_size': 512,
        'attention_size': 32,
        'hidden_units': [128, 32],
        'config': conf,
    }

def get_tensor():
    graph = tf.get_default_graph()

    doc_indices = graph.get_tensor_by_name('sparse_data_tensor/IteratorGetNext:1')
    doc_values = graph.get_tensor_by_name('sparse_data_tensor/IteratorGetNext:3')
    doc_length = graph.get_tensor_by_name('sparse_data_tensor/IteratorGetNext:2')
    #scale = graph.get_tensor_by_name('self_cosine_loss/weight_1:0')
    #bias = graph.get_tensor_by_name('self_cosine_loss/b_1:0')
    #print ('scale:', sess.run(scale))
    #print ('bias:', sess.run(bias))
    # image = tf.placeholder(tf.float32,[1,192,192,3])
    # print(image)
    output = graph.get_tensor_by_name('self_cosine_loss/add:0')
    return doc_indices, doc_values, doc_length

def default_input():
    doc_indices = tf.placeholder(tf.int64, [None, 2], 'doc_indices')
    doc_values = tf.placeholder(tf.int32, [None], 'doc_values')
    doc_length = tf.placeholder(tf.int32, [None], 'doc_length')
    return doc_indices, doc_values, doc_length

def smart_input(token_ids):
    #doc_length = token_ids.get_shape()
    doc_length = tf.shape(token_ids)
    doc_length = tf.identity(doc_length, 'doc_length')
    zero = tf.zeros(1, dtype=tf.int32)
    doc_values = tf.concat([token_ids, zero], 0, 'doc_values')
    #doc_values = tf.pad(token_ids, [[0, 1],[0,0]], name='doc_values')
    doc_indices = tf.constant([[i, 0] for i in range(128)], tf.int64)
    doc_indices = doc_indices[0:doc_length[0]]
    expand = tf.constant([[64, 0]], tf.int64)
    doc_indices = tf.concat([doc_indices, expand], 0, 'doc_indices')
    return doc_indices, doc_values, doc_length


# for export
export_path = r'w6_7_export'
init_op = tf.global_variables_initializer()


config = tf.ConfigProto()
# occupy gpu gracefully
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    with tf.device('device:CPU:0'):

        # load graph
        
        # option 1: from meta file
        #saver = tf.train.import_meta_graph(checkpoint+'.meta')
        
        # option 2: from predefined net
        token_ids = tf.placeholder(tf.int32, [None], 'token_ids')
        doc_indices, doc_values, doc_length = smart_input(token_ids)

        doc_net = model(doc_indices, doc_values, doc_length, get_params())
        saver = tf.train.Saver()

        # load values
        saver.restore(sess, checkpoint)
        
        tensors = {
            'token_ids': token_ids,
            'doc_net': doc_net
        }

        info = {name: utils.build_tensor_info(tensor) for name, tensor in tensors.items()}
        input_names = ['token_ids']
        output_names = ['doc_net']
        
        prediction_signature = signature_def_utils.build_signature_def(
            inputs={name: info[name] for name in input_names},
            outputs ={name: info[name] for name in output_names},
            method_name=signature_constants.PREDICT_METHOD_NAME
        )
        
        legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')

        builder = saved_model_builder.SavedModelBuilder(export_path)
        builder.add_meta_graph_and_variables(
            sess,
            [tag_constants.SERVING],
            signature_def_map={
                signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: prediction_signature,
            },
            legacy_init_op=legacy_init_op
        )
        builder.save(as_text=True)
        os.rename(os.path.join(export_path, 'saved_model.pbtxt'), os.path.join(export_path, export_path + '.saved_model.pbtxt'))
        os.rename(os.path.join(export_path, 'variables', 'variables.data-00000-of-00001'), os.path.join(export_path, export_path + '.variables.data-00000-of-00001'))
        os.rename(os.path.join(export_path, 'variables', 'variables.index'), os.path.join(export_path, export_path + '.variables.index'))
        os.rmdir(os.path.join(export_path, 'variables'))
print ('done')