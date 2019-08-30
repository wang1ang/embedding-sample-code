import argparse
import tensorflow as tf
import os
import codecs
import numpy as np
import data
import collections
import re
from datafiles import get_files
#from loss import self_cosine_loss, w2v_loss, mnlm_loss, order_loss
from cdssm import cdssm_tower, bert
#from data import SparseDataFeeder
from content import content_net
import config

conf = config.b1
init_checkpoint = r'bert\xlm_bert_convert_dis_query_layer3\best_model.ckpt'
lang_tier = 0 if not 'lang_tier' in conf else conf['lang_tier']
stepvalue = 500000

def model(features, labels, mode, params):
    batch_size = params['batch_size']
    doc_max_length = params['doc_max_length']
    embed_dim = params['embed_dim']
    conf = params['config']

    if 'doc_indices' in features:
        doc = tf.SparseTensor(indices=features['doc_indices'], values=features['doc_values'], dense_shape=(batch_size * doc_max_length, 64))
        doc_net = cdssm_tower(mode, doc, features['doc_length'], doc_max_length, embed_dim, 'doc', params, conf['activation'])
    else:
        bert_config = r'bert\xlm_bert_convert_dis_query_layer3\xlm_config_dis.json'
        doc_net = bert(bert_config, mode, params['hidden_units'][-1], features['doc_ids'], features['doc_mask'], features['doc_type'], conf['activation'],
            init_checkpoint)
    con_net = content_net(features['content'], doc_net.shape[-1], mode, conf['activation'])
    tf.summary.histogram('doc_net', doc_net)
    tf.summary.histogram('con_net', con_net)
    
    # l2-normalize
    if conf['query_l2']:
        doc_net = tf.nn.l2_normalize(doc_net, axis=1, epsilon=1e-3, name='l2_normalize_query')
    if conf['con_l2']:
        con_net = tf.nn.l2_normalize(con_net, axis=1, epsilon=1e-3, name='l2_normalize_content')

    # similarity matrix
    similarities = conf['sfunc'](doc_net, con_net)
        
    # loss
    loss, similarity, accuracy, loss_name = conf['loss'](mode, params, similarities, conf['version'])

    total_loss = loss + tf.add_n(tf.losses.get_regularization_losses())
    
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'url': features['url'],
            'cos_sim': similarity,
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    tf.summary.scalar(loss_name, loss)
    tf.summary.scalar("accuracy", accuracy)

    if mode == tf.estimator.ModeKeys.EVAL:
        metrics = {loss_name: tf.metrics.mean(loss), 'accuracy': tf.metrics.mean(accuracy)}
        return tf.estimator.EstimatorSpec(mode, loss=total_loss, eval_metric_ops=metrics)

    if params['init_checkpoint']:
        tvars = tf.trainable_variables()
        (assignment_map, initialized_variable_names) = get_assignment_map_from_checkpoint(tvars, params['init_checkpoint'])
        tf.train.init_from_checkpoint( params['init_checkpoint'], assignment_map)
    # Create training op.
    assert mode == tf.estimator.ModeKeys.TRAIN
    with tf.variable_scope('train_op'):
        global_step = tf.train.get_global_step()
        learning_rate = tf.train.exponential_decay(learning_rate=params['starter_learning_rate'], global_step=global_step, decay_steps=params['stepvalue'], decay_rate=params['gamma'], staircase=True)
        if 'optimizer' in conf:
            optimizer = conf['optimizer'](learning_rate)
        else:
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        train_op = optimizer.minimize(total_loss, global_step=global_step)
    tf.summary.scalar("learning_rate", learning_rate)
    return tf.estimator.EstimatorSpec(mode, loss=total_loss, train_op=train_op)

def main(argv):
    params = {
        #'vocab_size': len(vocab),
        'batch_size': 128,
        "eval_iter" : 100, # eval_iter * batch_size samples for testing
        "eval_interval" : 10000, # test every eval_interval iterations
        'input_max_length': 16, # query max length
        'doc_max_length': 64, # doc max length
        'embed_dim': 128,
        'rnn_size': 512,
        'attention_size': 32,
        'hidden_units': [128, 32],
        'train_log_interval': 1000,
        'starter_learning_rate' : 0.001 if 'starter_learning_rate' not in conf else conf['starter_learning_rate'], #0.01, # 0.01 -> 0.02
        'gamma' : 0.1,
        'stepvalue' : stepvalue, #500000,
        'num_train_steps' : 1500000,
        'init_checkpoint': None,
        'config': conf,
        'model_dir': 'models/' + conf['version'] +'/' # + '_0.02/'
    }

    # train
    #train_files = [['train_random_title_04{}_multi.txt'.format(n) for n in [13,15,16,17,18,19,20,21]]]
    train_files = get_files(lang_tier)
    #train_files = train_files[0:1]
    #test_files = ['train_random_title_0421_en.txt']
    test_files = [['train_HRS_title_April.txt']]
    embed_func = conf['embed_func']
    data_feeder = embed_func.get_feeder()
    train_feeder = data_feeder(
        params['batch_size'],
        train_files,
        params['doc_max_length'],
        skip_count=0,
        embed_func=embed_func,
        training = True)
    test_feeder = data_feeder(
        params['batch_size'],
        test_files,
        params['doc_max_length'],
        skip_count=0,
        embed_func=embed_func,
        training = False)
    params['vocab_size'] = train_feeder.get_vocab_size()

    session_config = tf.ConfigProto()
    session_config.gpu_options.allow_growth = True
    run_config = tf.estimator.RunConfig().replace(session_config=session_config)

    classifier = tf.estimator.Estimator(
        model_fn=model,
        params=params,
        model_dir=params['model_dir'], 
        config=run_config)

    train_spec = tf.estimator.TrainSpec(input_fn=lambda: [train_feeder.data_tensor(), None], max_steps=params['num_train_steps'])
    test_spec = tf.estimator.EvalSpec(input_fn=lambda: [test_feeder.data_tensor(), None], throttle_secs=600)
    tf.estimator.train_and_evaluate(classifier, train_spec, test_spec)

if __name__ == '__main__':

    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
