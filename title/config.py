import tensorflow as tf
from loss import *
v0 = {
    'version': 'v0',
    'activation': tf.nn.relu,
    'query_l2': True,
    'con_l2': False,
    'sfunc': dot, # order_Sfunc
    'loss': softmax_loss
}

v1 = {
    'version': 'v1',
    'activation': tf.nn.relu,
    'query_l2': True,
    'con_l2': True,
    'sfunc': dot, # order_Sfunc
    'loss': softmax_loss
}
v2 = {
    'version': 'v2',
    'activation': tf.nn.tanh,
    'query_l2': True,
    'con_l2': True,
    'sfunc': dot, # order_Sfunc
    'loss': softmax_loss
}

v3 = {
    'version': 'v3',
    'activation': None,
    'query_l2': True,
    'con_l2': True,
    'sfunc': dot, # order_Sfunc
    'loss': softmax_loss
}
v4 = {
    'version': 'v4',
    'activation': None,
    'query_l2': False,
    'con_l2': False,
    'sfunc': dot, # order_Sfunc
    'loss': w2v_loss
}
v5 = {
    'version': 'v5',
    'activation': None,
    'query_l2': False,
    'con_l2': False,
    'sfunc': dot, # order_Sfunc
    'loss': softmax_loss
}
v6 = {
    'version': 'v6',
    'activation': tf.nn.softplus,
    'query_l2': True,
    'con_l2': True,
    'sfunc': dot, # order_Sfunc
    'loss': softmax_loss
}
v6_1 = {
    'version': 'v6.1',
    'activation': tf.nn.softplus,
    'query_l2': False,
    'con_l2': False,
    'sfunc': dot, # order_Sfunc
    'loss': softmax_loss
}
v7 = {
    'version': 'v7',
    'activation': tf.abs,
    'query_l2': True,
    'con_l2': True,
    'sfunc': dot, # order_Sfunc
    'loss': softmax_loss
}
v8 = {
    'version': 'v8',
    'activation': None,
    'query_l2': True,
    'con_l2': True,
    'sfunc': dot, # order_Sfunc
    'loss': margin_loss
}
v9 = {
    'version': 'v9',
    'activation': tf.nn.softplus,
    'query_l2': False,
    'con_l2': False,
    'sfunc': order_Sfunc,
    'loss': margin_loss
}
v9_1 = {
    'version': 'v9.1',
    'activation': tf.nn.softplus,
    'query_l2': True,
    'con_l2': True,
    'sfunc': order_Sfunc,
    'loss': margin_loss
}
v10 = {
    'version': 'v10',
    'activation': None,
    'query_l2': False,
    'con_l2': False,
    'sfunc': relative_Sfunc, # order_Sfunc
    'loss': softmax_loss
}
print (v7['sfunc'] == dot)