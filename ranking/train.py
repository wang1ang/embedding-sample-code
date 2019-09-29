import tensorflow as tf
from data import DataFeeder
import os
from ndcg import get_ndcg
def rank_loss(r, s):
    with tf.variable_scope('rank_loss'):
        # ground truth
        #r_trans = tf.reshape(r, [1, -1])
        #r = tf.reshape(r, [-1, 1])
        r_col = tf.expand_dims(r, 1)
        r_row = tf.expand_dims(r, 2)
        p = tf.sign(r_col - r_row)
        # score
        #s_trans = tf.reshape(s, [1, -1])
        #s = tf.reshape(s, [-1, 1])
        s_col = tf.expand_dims(s, 1)
        s_row = tf.expand_dims(s, 2)
        diff = s_col - s_row
        # loss
        log_loss = tf.math.softplus(-(p * diff))
        #log_loss = tf.nn.relu(-(p * diff))
        log_loss = tf.reduce_mean(log_loss)
    tf.summary.scalar('ranking_loss', log_loss)
    return log_loss
def regression_loss(r, s, loss_name):
    with tf.variable_scope('regression_loss'):
        maes = tf.losses.absolute_difference(r, s)
        maes_loss = tf.reduce_mean(maes)
    tf.summary.scalar('label_loss', maes_loss)
    return maes_loss
def model(features, labels, mode, params):
    x = features['features']
    r = features['rating']
    # tf.nn.tanh
    # name=scope
    def bucket(x):
        s = tf.layers.dense(x, units=4, activation=tf.nn.tanh, kernel_initializer=tf.truncated_normal_initializer(stddev=0.0001), kernel_regularizer=tf.contrib.layers.l2_regularizer(0.004), name='layer_2')
        weights = tf.get_default_graph().get_tensor_by_name(os.path.split(s.name)[0] + '/kernel:0')
        s1, s2, s3, s4 = tf.split(s, 4, -1)
        s2 = tf.minimum(s1, s2)
        s3 = tf.minimum(s2, s3)
        s4 = tf.minimum(s3, s4)
        s = s1 + s2 + s3 + s4
        return s, weights
    def dense(x):
        s = tf.layers.dense(x, units=1, activation=None, kernel_initializer=tf.truncated_normal_initializer(stddev=0.0001), kernel_regularizer=tf.contrib.layers.l2_regularizer(0.004), name='layer_2')
        weights = tf.get_default_graph().get_tensor_by_name(os.path.split(s.name)[0] + '/kernel:0')
        return s, weights
    with tf.variable_scope('net'):
        # tf.initializers.glorot_normal()
        # x = tf.nn.batch_normalization(x)
        if (mode == tf.estimator.ModeKeys.TRAIN):
            noise_shape = tf.concat([tf.shape(x)[:-1], tf.ones(1, dtype=tf.int32)], 0)
            x = tf.nn.dropout(x, keep_prob=0.9, noise_shape=noise_shape)
        #x = tf.layers.dense(x, units=128, activation=tf.nn.tanh, kernel_initializer=tf.truncated_normal_initializer(stddev=0.001), kernel_regularizer=tf.contrib.layers.l2_regularizer(0.004), name='layer_1')
        s, weights = dense(x)
        #s, weights = bucket(x)
        s = tf.squeeze(s, axis=-1)
    tf.summary.histogram('weights', weights)
    tf.summary.histogram('score', s)

    loss = rank_loss(r, s)
    #loss = regression_loss(r, s, 'label_loss')
    total_loss = loss + tf.add_n(tf.losses.get_regularization_losses())

    #s = tf.nn.relu(s)
    #gains_loss = regression_loss(6*tf.pow(2., tf.nn.relu(r)), 6*tf.pow(2., tf.nn.relu(s)), 'my_loss')

    ndcg = get_ndcg(s, r)
    ndcg = tf.reduce_mean(ndcg)
    tf.summary.scalar('ndcg', ndcg)
    if mode == tf.estimator.ModeKeys.EVAL:
        metrics = {
            #'my_loss': tf.metrics.mean(gains_loss),
            #'label_loss': tf.metrics.mean(loss),
            'ranking_loss': tf.metrics.mean(loss),
            'ndcg': tf.metrics.mean(ndcg)
        } #, 'accuracy': tf.metrics.mean(accuracy)
        return tf.estimator.EstimatorSpec(mode, loss=total_loss, eval_metric_ops=metrics)

    with tf.variable_scope('train_op'):
        global_step = tf.train.get_global_step()
        learning_rate = tf.train.exponential_decay(learning_rate=params['starter_learning_rate'], global_step=global_step, decay_steps=params['stepvalue'], decay_rate=params['gamma'], staircase=True)
        #optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        #optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)
        optimizer = tf.train.AdamOptimizer(learning_rate)

        train_op = optimizer.minimize(total_loss, global_step=global_step)
    tf.summary.scalar("learning_rate", learning_rate)
    return tf.estimator.EstimatorSpec(mode, loss=total_loss, train_op=train_op)

def main(argv):
    params = {
        'model_dir': 'models/bingsat_rank_dense',
        'starter_learning_rate': 0.0001,
        'stepvalue': 20000,
        'gamma': 0.1,
        'num_train_steps': 40000,
    }
    files = ['data\\head_{}'.format(i) for i in range(1, 6)]
    with open('top_feature_list.txt', 'r') as f:
        top_features = [line.rstrip() for line in f.readlines()]
    data = DataFeeder(files, params['model_dir'])#, top_features)

    session_config = tf.ConfigProto()
    session_config.gpu_options.allow_growth = True
    run_config = tf.estimator.RunConfig().replace(session_config=session_config)

    classifier = tf.estimator.Estimator(
        model_fn=model,
        params=params,
        model_dir=params['model_dir'], 
        config=run_config)

    train_spec = tf.estimator.TrainSpec(input_fn=lambda: [data.data_tensor(True), None], max_steps=params['num_train_steps'])
    test_spec = tf.estimator.EvalSpec(input_fn=lambda: [data.data_tensor(False), None], throttle_secs=600)
    tf.estimator.train_and_evaluate(classifier, train_spec, test_spec)

if __name__ == '__main__':

    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
