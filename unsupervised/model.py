import tensorflow as tf
from datafeeder import DataFeeder
def hidden_layer(x, dim, scope):
    return tf.layers.dense(x, units=dim, activation=tf.nn.tanh, kernel_initializer=tf.initializers.glorot_normal(), kernel_regularizer=tf.contrib.layers.l2_regularizer(0.0004), name=scope) #tf.keras.initializers.he_normal()
def output_layer(x, dim, scope):
    return tf.layers.dense(x, units=dim, activation=tf.nn.relu, kernel_initializer=tf.initializers.glorot_normal(), kernel_regularizer=tf.contrib.layers.l2_regularizer(0.00004), name=scope) #tf.keras.initializers.he_normal()

def my_model(features, labels, mode, params):
    # content
    layer1 = hidden_layer(features[3], 4096, 'fc/content_fc1')
    layer2 = hidden_layer(layer1, 1024, 'fc/content_fc2')
    if mode==tf.estimator.ModeKeys.TRAIN:
        content = tf.nn.dropout(layer2, keep_prob=0.7)
    else:
        content = layer2

    # dan
    pred_dan = output_layer(content, params.dan_dim, scope='fc/fc_dan')

    pred_cdssm = output_layer(content, params.cdssm_dim, scope='fc/fc_cdssm')
    
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=tf.concat(1, [pred_dan, pred_cdssm]))

    dan = features[1]
    cdssm = features[2]
    # cdssm cosine
    cdssm_norm = tf.nn.l2_normalize(cdssm, axis=1)
    pred_cdssm_norm = tf.nn.l2_normalize(pred_cdssm, axis=1)
    loss_cdssm_cos = 1-tf.reduce_mean(tf.reduce_sum(cdssm_norm * pred_cdssm_norm, axis=1))

    loss_dan = tf.losses.mean_squared_error(dan, pred_dan) / 2.2
    loss_cdssm = tf.losses.mean_squared_error(cdssm, pred_cdssm) / 3000
    auto_regularization = tf.losses.get_regularization_losses()
    total_loss = loss_dan / 10 + loss_cdssm + loss_cdssm_cos + tf.add_n(auto_regularization) # loss_dan's variance is very high, so balance it by /100 (changed to /10 in v8)

    tf.summary.scalar('loss', total_loss)
    tf.summary.scalar('loss_dan', loss_dan)
    tf.summary.scalar('loss_cdssm', loss_cdssm)
    tf.summary.scalar('loss_cdssm_cos', loss_cdssm_cos)
    if mode == tf.estimator.ModeKeys.EVAL:
        metrics = {'loss_dan': tf.metrics.mean(loss_dan), 'loss_cdssm': tf.metrics.mean(loss_cdssm), 'loss_cdssm_cos': tf.metrics.mean(loss_cdssm_cos)}
        return tf.estimator.EstimatorSpec(mode, loss=total_loss, eval_metric_ops=metrics) # check loss of cdsssm
    else:
        global_step = tf.train.get_or_create_global_step()
        #global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(0.005, global_step, 100000, 0.9, staircase=True)
        tf.summary.scalar('learning_rate', learning_rate)
        #optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
        train_op = optimizer.minimize(total_loss, global_step=global_step) # tf.train.get_global_step()
        return tf.estimator.EstimatorSpec(mode, loss=total_loss, train_op=train_op)

def main(argv):
    session_config = tf.ConfigProto()
    session_config.gpu_options.allow_growth = True
    run_config = tf.estimator.RunConfig().replace(session_config=session_config)

    train_feeder = DataFeeder(['train_random_video.txt'] + ['train_random_video_04{}.txt'.format(n) for n in [13,15,16,17,18,19,20,21]])
    test_feeder = DataFeeder(['train_HRS_video.txt'], is_training=False)

    classifier = tf.estimator.Estimator(model_fn=my_model, model_dir='log8_moredata_cos', config=run_config, params=train_feeder) #
    train_spec = tf.estimator.TrainSpec(input_fn=lambda: [train_feeder.tensor_data_generator(128, 100), None])
    test_spec = tf.estimator.EvalSpec(input_fn=lambda: [test_feeder.tensor_data_generator(128, 100), None], throttle_secs=600)
    tf.estimator.train_and_evaluate(classifier, train_spec, test_spec)

if __name__ == '__main__':

    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
