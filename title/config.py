import tensorflow as tf
from loss import *
from data import SentencePieceEmbedding, TriLetterEmbedding, XlmEmbedding
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
w6_1 = {
    'version': 'w6.1',
    'embed_func': SentencePieceEmbedding,
    'activation': tf.nn.softplus,
    'query_l2': False,
    'con_l2': False,
    'sfunc': dot,
    'loss': softmax_loss
}
w6_2 = {
    'version': 'w6.2',
    'embed_func': SentencePieceEmbedding,
    'activation': tf.nn.softplus,
    'query_l2': False,
    'con_l2': False,
    'sfunc': dot,
    'loss': softmax_loss
}
w6_3 = {
    'version': 'w6.3',
    'embed_func': SentencePieceEmbedding,
    'activation': tf.nn.softplus,
    'query_l2': False,
    'con_l2': False,
    'sfunc': dot, # order_Sfunc
    'loss': softmax_loss
}
w6_4 = {
    'version': 'w6.adam',
    'embed_func': SentencePieceEmbedding,
    'activation': tf.nn.softplus,
    'query_l2': False,
    'con_l2': False,
    'sfunc': dot, # order_Sfunc
    'loss': softmax_loss,
    'optimizer': tf.train.AdamOptimizer
}
w6_5 = {
    'version': 'w6.multi',
    'lang_tier': 1,
    'embed_func': SentencePieceEmbedding,
    'activation': tf.nn.softplus,
    'query_l2': False,
    'con_l2': False,
    'sfunc': dot, # order_Sfunc
    'loss': softmax_loss,
    'optimizer': tf.train.AdamOptimizer
}
w6_6 = {
    # content: last layer 0.0004
    # top24_10w.model
    'version': 'w6.top24',
    'embed_func': SentencePieceEmbedding,
    'activation': tf.nn.softplus,
    'query_l2': False,
    'con_l2': False,
    'sfunc': dot, # order_Sfunc
    'loss': softmax_loss,
    'optimizer': tf.train.AdamOptimizer
}
w6_7 = {
    # content: last layer 0.0004
    # top24_10w.model
    # english training
    'version': 'w6.l2',
    'embed_func': SentencePieceEmbedding,
    'activation': tf.nn.softplus,
    'query_l2': True,
    'con_l2': True,
    'sfunc': dot, # order_Sfunc
    'loss': softmax_loss,
    'optimizer': tf.train.AdamOptimizer
}
w6_8 = {
    # content: last layer 0.0004
    # top24_10w.model
    # multi lan training
    'version': 'w6.l2_multi',
    'lang_tier': 1,
    'embed_func': SentencePieceEmbedding,
    'activation': tf.nn.softplus,
    'query_l2': True,
    'con_l2': True,
    'sfunc': dot, # order_Sfunc
    'loss': softmax_loss,
    'optimizer': tf.train.AdamOptimizer
}
w6_9 = {
    # content: last layer 0.0004
    # top24_10w.model
    # multi lan training
    'version': 'w6.l2_multi_lr',
    'lang_tier': 1,
    'embed_func': SentencePieceEmbedding,
    'activation': tf.nn.softplus,
    'query_l2': True,
    'con_l2': True,
    'sfunc': dot, # order_Sfunc
    'loss': softmax_loss,
    'optimizer': tf.train.AdamOptimizer,
    'starter_learning_rate': 0.0001,
}

v6_2 = {
    'version': 'v6.2', # retry
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
v9_3 = {
    'version': 'v9.3',
    'activation': tf.nn.softplus,
    'query_l2': True,
    'con_l2': True,
    'sfunc': order_Sfunc,
    'loss': softmax_loss
}
v10 = {
    'version': 'v10_fix',
    'activation': None,
    'query_l2': False,
    'con_l2': False,
    'sfunc': relative_Sfunc, # order_Sfunc
    'loss': softmax_loss
}
v11 = {
    'version': 'v11',
    'activation': tf.nn.softplus,
    'query_l2': False,
    'con_l2': False,
    'sfunc': inter_Sfunc, # order_Sfunc
    'loss': softmax_loss
}
v11_1 = {
    'version': 'v11.1',
    'activation': None,
    'query_l2': False,
    'con_l2': False,
    'sfunc': inter_Sfunc, # order_Sfunc
    'loss': softmax_loss
}
#print (v7['sfunc'] == dot)

b0 = {
    # content: last layer 0.0004
    # multi lan training
    'version': 'b0',
    'lang_tier': 0,
    'embed_func': XlmEmbedding,
    'activation': tf.nn.softplus,
    'query_l2': True,
    'con_l2': True,
    'sfunc': dot, # order_Sfunc
    'loss': softmax_loss,
    'optimizer': tf.train.AdamOptimizer,
    'starter_learning_rate': 0.0001,
}

b1 = {
    # content: last layer 0.0004
    # multi lan training
    # init from xml params
    'version': 'b1',
    'lang_tier': 0,
    'embed_func': XlmEmbedding,
    'activation': tf.nn.softplus,
    'query_l2': True,
    'con_l2': True,
    'sfunc': dot, # order_Sfunc
    'loss': softmax_loss,
    'optimizer': tf.train.AdamOptimizer,
    'starter_learning_rate': 0.0001,
}

b1_1 = {
    # content: last layer 0.0004
    # multi lan training
    'version': 'b1_multi',
    'lang_tier': 1,
    'embed_func': XlmEmbedding,
    'activation': tf.nn.softplus,
    'query_l2': True,
    'con_l2': True,
    'sfunc': dot, # order_Sfunc
    'loss': softmax_loss,
    'optimizer': tf.train.AdamOptimizer,
    'starter_learning_rate': 0.0001,
}

b1_2 = {
    # content: last layer 0.0004
    # multi lan training
    # merge chs & chz
    'version': 'b1_multi_more',
    'lang_tier': 1,
    'embed_func': XlmEmbedding,
    'activation': tf.nn.softplus,
    'query_l2': True,
    'con_l2': True,
    'sfunc': dot, # order_Sfunc
    'loss': softmax_loss,
    'optimizer': tf.train.AdamOptimizer,
    'starter_learning_rate': 0.0001,
}
b1_3 = {
    # content: last layer 0.0004
    # multi lan training
    # merge chs & chz
    'version': 'b1_multi_2test',
    'lang_tier': 1,
    'embed_func': XlmEmbedding,
    'activation': tf.nn.softplus,
    'query_l2': True,
    'con_l2': True,
    'sfunc': dot, # order_Sfunc
    'loss': softmax_loss,
    'optimizer': tf.train.AdamOptimizer,
    'starter_learning_rate': 0.0001,
}

r1 = {
    # regression
    'version': 'r1',
    'lang_tier': 1,
    'embed_func': XlmEmbedding,
    'activation': tf.nn.softplus,
    'query_l2': True,
    'con_l2': True,
    'sfunc': dot, # order_Sfunc
    'loss': softmax_loss,
    'optimizer': tf.train.AdamOptimizer,
    'starter_learning_rate': 0.0001,
}
