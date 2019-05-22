import argparse
import tensorflow as tf
import os
import codecs
import numpy as np
import data_prepare

parser = argparse.ArgumentParser()
parser.add_argument('--mode', default="train", help='train or evaluate')
parser.add_argument('--model_dir', default="model_test/v10_newfeeder/", help='model file dir')
parser.add_argument('--train_file', default="AllData_part1_2018_07_27_top1M.tsv", help='training data file')
parser.add_argument('--test_file', default="AllData_part1_2018_07_27_top1M_top128k.tsv", help='testing data file')
parser.add_argument('--predict_file', default="./PredictDataSample.tsv", help='predict data file')
parser.add_argument('--predict_file_result', default="./PredictDataSampleResult.tsv", help='predict data result file')
parser.add_argument('--query_col', default=1, type=int, help='query column index')
parser.add_argument('--doc_col', default=2, type=int, help='document column index')
parser.add_argument('--rawquery_col', default=1, type=int, help='rawquery column index')
parser.add_argument('--rawurl_col', default=0, type=int, help='url column index')
parser.add_argument('--batch_size', default=128, type=int, help='batch size')
#parser.add_argument('--train_steps', default=1000, type=int, help='number of training steps')
parser.add_argument('--embedding_size', default=128, type=int, help='embedding size')
parser.add_argument('--num_train_steps', default=1000000, type=int, help='number of max iterations')
parser.add_argument('--starter_learning_rate', default=0.15, type=float, help='start learning rate')
parser.add_argument('--gamma', default=0.1, type=float, help='learning rate decay gamma')
parser.add_argument('--stepvalue', default=300000, type=int, help='learning rate decay every k steps')

def sparse_embed_layer(sparse_input, vocab_size, embed_dim):
    with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
        #embedding_table = tf.get_variable("embedding_table", [vocab_size, embed_dim], dtype=tf.float32, initializer=tf.random_normal_initializer(), regularizer=tf.contrib.layers.l2_regularizer(0.00004))
        embedding_table = tf.get_variable("embedding_table", [vocab_size, embed_dim], dtype=tf.float32, initializer=tf.random_normal_initializer())
        #embedding_table = tf.get_variable("embedding_table", [vocab_size, embed_dim], dtype=tf.float32, initializer=tf.random_uniform_initializer())
        return tf.nn.embedding_lookup_sparse(embedding_table, sparse_input, sp_weights=None, combiner='sum')

def cdssm_tower(feature, length, max_length, embed_dim, scope, params):
    batch_size = params['batch_size']
    vocab_size = params['vocab_size']
    embedding = sparse_embed_layer(feature, vocab_size, embed_dim)
    embedding = tf.slice(embedding, [0,0], [batch_size * max_length ,-1])
    embedding = tf.reshape(embedding, shape=(batch_size, max_length, embed_dim))
    rnn_layers = [tf.nn.rnn_cell.LSTMCell(512, activation=tf.tanh)]
    multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)
    outputs, states = tf.nn.dynamic_rnn(multi_rnn_cell, inputs=embedding, sequence_length=length, dtype=tf.float32, scope=scope)
    last_state = states[-1][-1]
    # fully connected layers
    for i, units in enumerate(params['hidden_units']):
        fc = tf.layers.dense(last_state, units=units, activation=tf.nn.tanh, kernel_initializer=tf.initializers.glorot_normal(), kernel_regularizer=tf.contrib.layers.l2_regularizer(0.0004), name=scope+'_fc'+str(i)) #tf.keras.initializers.he_normal()
    norm = tf.nn.l2_normalize(fc, axis=1, epsilon=1e-3)
    return norm

def self_cosine_loss(mode, hparams, query, doc):
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

def cdssm_model(features, labels, mode, params):
    batch_size = params['batch_size']
    input_max_length = params['input_max_length']
    doc_max_length = params['doc_max_length']
    embed_dim = params['embed_dim']
    inp = tf.SparseTensor(indices=features['input_indices'], values=features['input_values'], dense_shape=(batch_size * input_max_length, 64))
    doc = tf.SparseTensor(indices=features['doc_indices'], values=features['doc_values'], dense_shape=(batch_size * doc_max_length, 64))
    query_net = cdssm_tower(inp, features['input_length'], input_max_length, embed_dim, 'query', params)
    doc_net = cdssm_tower(doc, features['doc_length'], doc_max_length, embed_dim, 'doc', params)

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

    # Create training op.
    assert mode == tf.estimator.ModeKeys.TRAIN
    global_step = tf.train.get_global_step()
    learning_rate = tf.train.exponential_decay(learning_rate=params['starter_learning_rate'], global_step=global_step, decay_steps=params['stepvalue'], decay_rate=params['gamma'], staircase=True)
    tf.summary.scalar("learning_rate", learning_rate)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.minimize(total_loss, global_step=global_step)
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
        'hidden_units': [128, 32],
        'train_log_interval': 1000,
        'starter_learning_rate' : args.starter_learning_rate,
        'gamma' : args.gamma,
        'stepvalue' : args.stepvalue,
        'num_train_steps' : args.num_train_steps,
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
        train_spec = tf.estimator.TrainSpec(input_fn=lambda: [train_feeder.data_tensor(), None], max_steps=args.num_train_steps)
        test_spec = tf.estimator.EvalSpec(input_fn=lambda: [test_feeder.data_tensor(), None], throttle_secs=60)
        tf.estimator.train_and_evaluate(classifier, train_spec, test_spec)

        """
        train_input_fn, train_feed_fn = data_prepare.make_input_fn(
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
        test_input_fn, test_feed_fn = data_prepare.make_input_fn(
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
        train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, hooks=[tf.train.FeedFnHook(train_feed_fn)], max_steps=args.num_train_steps)
        test_spec = tf.estimator.EvalSpec(input_fn=test_input_fn, hooks=[tf.train.FeedFnHook(test_feed_fn)] , throttle_secs=60)
        tf.estimator.train_and_evaluate(classifier, train_spec, test_spec)
        """
    else:
        # predict
        predict_input_fn, predict_feed_fn = data_prepare.make_input_fn(
            params['batch_size'],
            args.predict_file,
            query_col,
            doc_col,
            rawquery_col,
            rawurl_col,
            vocab, params['input_max_length'], 
            params['doc_max_length'], 
            skip_count=0, 
            shuffle = False,
            training = False)
        
        predictions = classifier.predict(
            input_fn=predict_input_fn,
            hooks=[tf.train.FeedFnHook(predict_feed_fn)]
            )
    
        eva_score_list = []
        eva_ecd_list = []
        query_list = []
        url_list = []

        count = 0
    
        score_list_stream = codecs.open(args.predict_file_result, "w", "utf-8")
        for pred_dict in predictions:

            query = pred_dict['query'].split('\n')
            url = pred_dict['url'].split('\n')
            cos_sim = str(pred_dict['cos_sim']).split('\n')
            euc_sim = str(pred_dict['euc_sim']).split('\n')
       
            count = count + 1
            if count % 1000 == 0:
                print (count)

            for i in range(len(query)):
                score_list_stream.write("{0}\t{1}\t{2}\t{3}\n".format(str(query[i]), str(url[i]),str(cos_sim[i]), str(euc_sim[i])))
        '''
        with open(args.predict_file_result, 'w') as score_list_stream:
            for i in range(len(query_list)):
                score_list_stream.write("{0}\t{1}\t{2}\t{3}\n".format(str(query_list[i]), str(url_list[i]),str(eva_score_list[i]), str(eva_ecd_list[i])))
            '''
        score_list_stream.close()

if __name__ == '__main__':

    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
