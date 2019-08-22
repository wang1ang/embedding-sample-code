import tensorflow as tf
def hidden_layer(x, dim, scope):
    return tf.layers.dense(x, units=dim, activation=tf.nn.tanh, kernel_initializer=tf.initializers.glorot_normal(), kernel_regularizer=tf.contrib.layers.l2_regularizer(0.0004), name=scope) #tf.keras.initializers.he_normal()
def output_layer(x, dim, scope, activation):
    return tf.layers.dense(x, units=dim, activation=activation, kernel_initializer=tf.initializers.glorot_normal(), kernel_regularizer=tf.contrib.layers.l2_regularizer(0.0004), name=scope) #tf.keras.initializers.he_normal()
def content_net(content, cdssm_dim, mode, activation):
    # content
    layer1 = hidden_layer(content, 4096, 'content/content_fc1')
    layer2 = hidden_layer(layer1, 1024, 'content/content_fc2')
    if mode==tf.estimator.ModeKeys.TRAIN:
        layer2 = tf.nn.dropout(layer2, keep_prob=0.7, name='content/dropout')
    layer3 = output_layer(layer2, cdssm_dim, 'content/content_fc3', activation)
    return layer3