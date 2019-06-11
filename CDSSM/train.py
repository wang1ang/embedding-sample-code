import argparse
import tensorflow as tf
import os
import codecs
import numpy as np
import data_prepare
import collections
import re

parser = argparse.ArgumentParser()
parser.add_argument('--mode', default="train", help='train or evaluate')
parser.add_argument('--model_dir', default="model_test/v15_64_lr001/", help='model file dir')
#parser.add_argument('--train_file', default="AllData_part1_2018_07_27_top1M.tsv", help='training data file')
#parser.add_argument('--test_file', default="AllData_part1_2018_07_27_top1M_top128k.tsv", help='testing data file')
parser.add_argument('--train_file', default="AllDataWithImage_full_norm_dedup_2018_07_27_noimage.tsv", help='training data file')
parser.add_argument('--test_file', default="AllDataWithImage_full_norm_dedup_2018_07_27_noimage_top128k.tsv", help='testing data file')
parser.add_argument('--init_checkpoint', default='model_test/v14_big/model.ckpt-960307', help='init checkpoint')

parser.add_argument('--predict_file', default="./PredictDataSample.tsv", help='predict data file')
parser.add_argument('--predict_file_result', default="./PredictDataSampleResult.tsv", help='predict data result file')

parser.add_argument('--query_col', default=2, type=int, help='query column index')
parser.add_argument('--doc_col', default=3, type=int, help='document column index')
parser.add_argument('--rawquery_col', default=2, type=int, help='rawquery column index')
parser.add_argument('--rawurl_col', default=1, type=int, help='url column index')

parser.add_argument('--batch_size', default=128, type=int, help='batch size')
#parser.add_argument('--train_steps', default=1000, type=int, help='number of training steps')
parser.add_argument('--embedding_size', default=128, type=int, help='embedding size')
parser.add_argument('--rnn_size', default=512, type=float, help='rnn size')
parser.add_argument('--attention_size', default=32, type=float, help='attention size')
parser.add_argument('--num_train_steps', default=1000000, type=int, help='number of max iterations')
parser.add_argument('--starter_learning_rate', default=0.01, type=float, help='start learning rate') # 0.15
parser.add_argument('--gamma', default=0.1, type=float, help='learning rate decay gamma')
parser.add_argument('--stepvalue', default=300000, type=int, help='learning rate decay every k steps')

def sparse_embed_layer(embedding_table, sparse_input, batch_size, max_length, embed_dim):
    with tf.variable_scope('embed_lookup'):
        embedding = tf.nn.embedding_lookup_sparse(embedding_table, sparse_input, sp_weights=None, combiner='sum')
        embedding = tf.slice(embedding, [0,0], [batch_size * max_length ,-1])
        embedding = tf.reshape(embedding, shape=(batch_size, max_length, embed_dim))
    return embedding
def rnn(embedding, length, rnn_size, scope):
    rnn_layers = [tf.nn.rnn_cell.LSTMCell(rnn_size)]
    multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)
    outputs, states = tf.nn.dynamic_rnn(multi_rnn_cell, inputs=embedding, sequence_length=length, dtype=tf.float32, scope=scope)
    return states[-1][-1]
def birnn(embedding, length, rnn_size):
    cell_fw = tf.nn.rnn_cell.LSTMCell(rnn_size)
    cell_bw = tf.nn.rnn_cell.LSTMCell(rnn_size)
    outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw, cell_bw=cell_bw, inputs=embedding, sequence_length=length, dtype=tf.float32, scope='birnn')
    output = tf.concat([outputs[0], outputs[1]], axis=2)
    last_state = tf.concat([states[0][-1], states[1][-1]], axis=1)
    return output, last_state
def attention(sequence, attention_size, scope):
    with tf.name_scope('attention'), tf.variable_scope(scope + '_attention'):
        attention = tf.layers.dense(sequence, units=attention_size, activation=tf.nn.tanh, kernel_initializer=tf.initializers.glorot_normal(), kernel_regularizer=tf.contrib.layers.l2_regularizer(0.0004), name=scope+'_attention_1')
        attention = tf.layers.dense(sequence, units=1, activation=None, kernel_initializer=tf.initializers.glorot_normal(), kernel_regularizer=tf.contrib.layers.l2_regularizer(0.0004), name=scope+'_attention_2')
        alpha = tf.nn.softmax(attention)
        output = sequence * alpha
        return tf.reduce_sum(output, 1)

def cdssm_tower(mode, feature, length, max_length, embed_dim, scope, params):
    batch_size = params['batch_size']
    vocab_size = params['vocab_size']
    #embedding = sparse_embed_layer(feature, vocab_size, embed_dim, scope)
    with tf.variable_scope('common', reuse=tf.AUTO_REUSE):
        embedding_table = tf.get_variable("embedding_table", [vocab_size, embed_dim], dtype=tf.float32, initializer=tf.random_normal_initializer())
    with tf.variable_scope(scope):
        embedding = sparse_embed_layer(embedding_table, feature, batch_size, max_length, embed_dim)
        outputs, last_state = birnn(embedding, length, params['rnn_size'])
        pooled = attention(outputs, params['attention_size'], scope)
        #pooled = last_state
        # fully connected layers
        fc = pooled
        for i, units in enumerate(params['hidden_units']):
            fc = tf.layers.dense(fc, units=units, activation=tf.nn.tanh, kernel_initializer=tf.initializers.glorot_normal(), kernel_regularizer=tf.contrib.layers.l2_regularizer(0.0004), name=scope+'_fc'+str(i)) #tf.keras.initializers.he_normal()
            #if i == 0 and mode==tf.estimator.ModeKeys.TRAIN:
            #    fc = tf.nn.dropout(fc, keep_prob=0.5)
        norm = tf.nn.l2_normalize(fc, axis=1, epsilon=1e-3)
        return norm

def self_cosine_loss(mode, hparams, query, doc):
    with tf.variable_scope('self_cosine_loss'):
        # suppose query & doc are l2-normalized
        cosines = tf.matmul(query, doc, transpose_b=True)
        
        # scale & softmax
        W = tf.get_variable('weight_1', [1], initializer=tf.constant_initializer(5.0))
        b = tf.get_variable('b_1', [1], initializer=tf.constant_initializer(0.01))
        prob = tf.nn.softmax(cosines * W + b, axis=1)
        
        # cross entropy loss
        diag = tf.linalg.tensor_diag_part(prob)
        loss = -tf.log(diag)
        loss = tf.reduce_mean(loss, name="trainloss")

        # measurements
        batchsize = hparams['batch_size']
        prediction = tf.argmax(cosines, axis=1, output_type=tf.int32)
        accuracy = tf.equal(prediction, tf.constant(np.arange(batchsize, dtype=np.int32), dtype=tf.int32))
        accuracy = tf.reduce_mean(tf.cast(accuracy, tf.float32))
        similarity = tf.linalg.tensor_diag_part(cosines)
        return loss, similarity, accuracy

def get_assignment_map_from_checkpoint(tvars, init_checkpoint):
  """Compute the union of the current variables and checkpoint variables."""
  assignment_map = {}
  initialized_variable_names = {}

  name_to_variable = collections.OrderedDict()
  for var in tvars:
    name = var.name
    m = re.match("^(.*):\\d+$", name)
    if m is not None:
      name = m.group(1)
    name_to_variable[name] = var

  init_vars = tf.train.list_variables(init_checkpoint)

  assignment_map = collections.OrderedDict()
  for x in init_vars:
    (name, shape) = (x[0], x[1])
    if name not in name_to_variable:
      print ('{}: not load'.format(name))
      continue
    if name_to_variable[name].shape != shape:
      print ('{}: not load, shape not match {} -> {}'.format(name, shape, name_to_variable[name].shape))
      continue
    assignment_map[name] = name
    initialized_variable_names[name] = 1
    initialized_variable_names[name + ":0"] = 1
  return (assignment_map, initialized_variable_names)

def cdssm_model(features, labels, mode, params):
    batch_size = params['batch_size']
    input_max_length = params['input_max_length']
    doc_max_length = params['doc_max_length']
    embed_dim = params['embed_dim']
    inp = tf.SparseTensor(indices=features['input_indices'], values=features['input_values'], dense_shape=(batch_size * input_max_length, 64))
    doc = tf.SparseTensor(indices=features['doc_indices'], values=features['doc_values'], dense_shape=(batch_size * doc_max_length, 64))
    query_net = cdssm_tower(mode, inp, features['input_length'], input_max_length, embed_dim, 'query', params)
    doc_net = cdssm_tower(mode, doc, features['doc_length'], doc_max_length, embed_dim, 'doc', params)

    # loss
    loss, similarity, accuracy  = self_cosine_loss(mode, params, query_net, doc_net)
    total_loss = loss + tf.add_n(tf.losses.get_regularization_losses())
    
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'query': features['query'],
            'url': features['url'],
            'cos_sim': similarity,
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    tf.summary.scalar("my_loss", loss)
    tf.summary.scalar("accuracy", accuracy)

    if mode == tf.estimator.ModeKeys.EVAL:
        metrics = {'my_loss': tf.metrics.mean(loss), 'accuracy': tf.metrics.mean(accuracy)}
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

        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        train_op = optimizer.minimize(total_loss, global_step=global_step)
    tf.summary.scalar("learning_rate", learning_rate)
    return tf.estimator.EstimatorSpec(mode, loss=total_loss, train_op=train_op)

def main(argv):
    args = parser.parse_args(argv[1:])
    vocab = data_prepare.trilettergram()

    params = {
        'vocab_size': len(vocab),
        'batch_size': args.batch_size,
        "eval_iter" : 100, # eval_iter * batch_size samples for testing
        "eval_interval" : 10000, # test every eval_interval iterations
        'input_max_length': 16, # query max length
        'doc_max_length': 64, # doc max length
        'embed_dim': args.embedding_size,
        'rnn_size': args.rnn_size,
        'attention_size': args.attention_size,
        'hidden_units': [128, 64],
        'train_log_interval': 1000,
        'starter_learning_rate' : args.starter_learning_rate,
        'gamma' : args.gamma,
        'stepvalue' : args.stepvalue,
        'num_train_steps' : args.num_train_steps,
        'init_checkpoint': args.init_checkpoint,
    }

    query_col = args.query_col
    doc_col = args.doc_col
    rawquery_col = args.rawquery_col
    rawurl_col = args.rawurl_col
    
    session_config = tf.ConfigProto()
    session_config.gpu_options.allow_growth = True
    run_config = tf.estimator.RunConfig().replace(session_config=session_config)

    classifier = tf.estimator.Estimator(
        model_fn=cdssm_model,
        params=params,
        model_dir=args.model_dir, 
        config=run_config)

    # train
    if (args.mode == "train"):
        train_feeder = data_prepare.DataFeeder(
            params['batch_size'],
            args.train_file,
            query_col,
            doc_col,
            rawquery_col,
            rawurl_col,
            vocab, params['input_max_length'], 
            params['doc_max_length'],
            skip_count=128000,
            training = True)
        test_feeder = data_prepare.DataFeeder(
            params['batch_size'],
            args.test_file,
            query_col,
            doc_col,
            rawquery_col,
            rawurl_col,
            vocab, params['input_max_length'], 
            params['doc_max_length'],
            skip_count=0,
            training = False)
        train_spec = tf.estimator.TrainSpec(input_fn=lambda: [train_feeder.sparse_data_tensor(), None], max_steps=args.num_train_steps)
        test_spec = tf.estimator.EvalSpec(input_fn=lambda: [test_feeder.sparse_data_tensor(), None], throttle_secs=60)
        tf.estimator.train_and_evaluate(classifier, train_spec, test_spec)

if __name__ == '__main__':

    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
