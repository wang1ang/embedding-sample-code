
import tensorflow as tf

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

def cdssm_tower(mode, feature, length, max_length, embed_dim, scope, params, activation):
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
            if i < len(params['hidden_units'])-1:
                fc = tf.layers.dense(fc, units=units, activation=tf.nn.tanh, kernel_initializer=tf.initializers.glorot_normal(), kernel_regularizer=tf.contrib.layers.l2_regularizer(0.0004), name=scope+'_fc'+str(i)) #tf.keras.initializers.he_normal()
            else:
                fc = tf.layers.dense(fc, units=units, activation=activation, kernel_initializer=tf.initializers.glorot_normal(), kernel_regularizer=tf.contrib.layers.l2_regularizer(0.0004), name=scope+'_fc'+str(i)) #tf.keras.initializers.he_normal()
            #if i == 0 and mode==tf.estimator.ModeKeys.TRAIN:
            #    fc = tf.nn.dropout(fc, keep_prob=0.5)
        return fc
