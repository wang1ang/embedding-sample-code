
import tensorflow as tf
from bert.modeling import BertModel, BertConfig, get_assignment_map_from_checkpoint

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
    # todo: try tf.random_normal_initializer(mean=0.0, stddev=0.02)
    # todo: try tf.random_uniform_initializer(minval=-0.02, maxval=0.02)
    # todo: try tf.truncated_normal_initializer(stddev=0.02) (used in bert) will be redrawn if the value > 2 x 0.02
    with tf.variable_scope('common', reuse=tf.AUTO_REUSE):
        embedding_table = tf.get_variable("embedding_table", [vocab_size, embed_dim], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
    with tf.variable_scope(scope):
        embedding = sparse_embed_layer(embedding_table, feature, batch_size, max_length, embed_dim)
        #embedding = tf.contrib.layers.layer_norm(inputs=embedding, begin_norm_axis=-1, begin_params_axis=-1, scope=name)
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
def bert(bert_config_file, mode, dim, input_ids, input_mask, input_type, activation, init_checkpoint=None):
    bert_config = BertConfig.from_json_file(bert_config_file)
    bert_model = BertModel(
        config=bert_config,
        is_training=mode==tf.estimator.ModeKeys.TRAIN,
        input_ids= input_ids,
        input_mask=input_mask,
        token_type_ids= input_type,
        scope="bert_query"
    )
    output = bert_model.get_pooled_output()
    if mode==tf.estimator.ModeKeys.TRAIN:
        output = tf.nn.dropout(output, keep_prob=0.9)
    sig = tf.layers.dense(output, dim, activation=activation, kernel_initializer=tf.truncated_normal_initializer(stddev=bert_config.initializer_range), name="bert_query/query")
    

    tvars = tf.trainable_variables('bert_query')
    initialized_variable_names = {}
    if init_checkpoint:
      (assignment_map, initialized_variable_names) = get_assignment_map_from_checkpoint(tvars, init_checkpoint)
      tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
    """
    for var in tvars:
      init_string = ""
      if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
      tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                      init_string)
    """

    return sig