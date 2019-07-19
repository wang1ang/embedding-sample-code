import tensorflow as tf
import numpy as np

def dot(x, y):
    return tf.matmul(x, y, transpose_b=True)

def similarity_accuracy(smatrix, size):
        prediction = tf.argmax(smatrix, axis=1, output_type=tf.int32)
        accuracy = tf.equal(prediction, tf.constant(np.arange(size, dtype=np.int32), dtype=tf.int32))
        accuracy = tf.reduce_mean(tf.cast(accuracy, tf.float32))
        similarity = tf.linalg.tensor_diag_part(smatrix)
        return similarity, accuracy

def _marginal_loss(smatrix, size, margin):
    pos, accuracy = similarity_accuracy(smatrix, size)
    # query x doc
    pos_row = tf.reshape(pos, [1, -1])
    pos_col = tf.reshape(pos, [-1, 1])
    loss = tf.maximum(margin - pos_row + smatrix, 0) + tf.maximum(margin - pos_col + smatrix, 0)
    loss = tf.linalg.set_diag(loss, tf.zeros(size, dtype=loss.dtype))
    loss = tf.reduce_mean(loss) # - (bs*2)
    return loss, pos, accuracy

def _softmax_loss(smatrix, size):
    with tf.variable_scope('softmax_loss'):
        prob = tf.nn.softmax(smatrix, axis=1)
        # cross entropy loss
        diag = tf.linalg.tensor_diag_part(prob)
        loss = tf.log(diag)
        loss = -tf.reduce_mean(loss, name="trainloss")

        # measurements
        similarity, accuracy = similarity_accuracy(smatrix, size)
        return loss, similarity, accuracy

def softmax_loss(mode, hparams, similarities, version):
    with tf.variable_scope('self_cosine_loss'):
        batchsize = hparams['batch_size']
        conf = hparams['config']
        if (conf['query_l2'] or conf['con_l2']) and conf['sfunc'] == relative_Sfunc:
            W = tf.get_variable('weight_1', [1], initializer=tf.constant_initializer(5.0))
            b = tf.get_variable('b_1', [1], initializer=tf.constant_initializer(0.01))
            similarities = similarities * W + b
        loss, similarity, accuracy = _softmax_loss(similarities, batchsize)
        return loss, similarity, accuracy, 'my_loss'


def w2v_loss(mode, hparams, similarities, version):
    # this loss is used in word2vec
    # it doesn't explicitly represent the accuracy
    # it gives bad accuracy in each batch
    bs = hparams['batch_size']
    with tf.variable_scope('w2v_loss'):
        
        #mask = np.identity(bs, np.int32) * 2 - 1
        mask = np.identity(bs, np.float32) - 1.0 / bs

        mask = tf.constant(mask, tf.float32)
        #prob = tf.sigmoid(similarities * mask)
        #logp = tf.log(prob)
        logp = tf.log_sigmoid(similarities*mask)
        loss = -tf.reduce_mean(logp, name="trainloss")

        similarity, accuracy = similarity_accuracy(similarities, bs)

        return loss, similarity, accuracy, "w2vloss"

def order_Efunc(x, y):
    # x: 128 x 32 -> 128 x 1 x 32
    # y: 128 x 32 -> 1 x 128 x 32
    x = tf.expand_dims(x, 1)
    y = tf.expand_dims(y, 0)
    diff = y - x
    m = tf.maximum(diff, 0)
    sqr = tf.square(m)
    E = tf.reduce_sum(sqr, axis=2)
    return E
def order_Sfunc(x, y):
    with tf.variable_scope('order_similarity'):
        return -order_Efunc(y,x)
def margin_loss(mode, hparams, similarities, version):
    # this loss is from "Order-embeddings of images and language"
    bs = hparams['batch_size']
    conf = hparams['config']
    if conf['sfunc'] == order_Sfunc:
        if conf['query_l2'] and conf['con_l2']:
            margin = 0.05
        else:
            margin = 1
    else:
        margin = 0.2
    with tf.variable_scope('margin_loss'):
        loss, pos, accuracy = _marginal_loss(similarities, bs, margin)
        return loss, pos, accuracy, 'marginloss'

def relative_Sfunc(query, doc):
    relative = doc - query
    query = tf.nn.l2_normalize(query, axis=1, epsilon=1e-3, name='l2_normalize_query')
    relative = tf.nn.l2_normalize(relative, axis=1, epsilon=1e-3, name='l2_normalize_relative')
    return dot(query, relative)
#def relative_cosine_loss(mode, hparams, query, doc):
#    with tf.variable_scope('relative_cosine_loss'):
#        return self_cosine_loss(mode, hparams, query, doc)