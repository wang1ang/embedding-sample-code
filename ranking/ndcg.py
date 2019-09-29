import tensorflow as tf
def get_approx_ranks(scores, alpha=10):
    """
    The rank of an item in the list is simply one plus the total number of items with a larger score. In other words,
      rank_i = 1 + \sum_{j \neq i} I_{s_j > s_i},
    where "I" is the indicator function. The indicator function can be approximated by a generalized sigmoid:
      I_{s_j < s_i} \approx 1/(1 + exp(-\alpha * (s_j - s_i))).
    This function approximates the rank of an item using this sigmoid approximation to the indicator function. 
    This technique is at the core of "A general approximation framework for direct optimization of information retrieval measures" by Qin et al.
    Args:
      logits: A `Tensor` with shape [batch_size, list_size]. Each value is the
        ranking score of the corresponding item.
      alpha: Exponent of the generalized sigmoid function.
    Returns:
      A `Tensor` of ranks with the same shape as logits.
    """
    x = tf.expand_dims(scores, 2)
    y = tf.expand_dims(scores, 1)
    pairs = tf.sigmoid(alpha * (y-x))
    return tf.reduce_sum(pairs, axis=-1) + .5

def get_ranks(scores):
    x = tf.expand_dims(scores, 2)
    y = tf.expand_dims(scores, 1)
    #pairs = tf.nn.relu(tf.sign(y-x))
    #return tf.reduce_sum(pairs, axis=-1) + 1
    pairs = tf.sign(y-x) + 1
    return (tf.reduce_sum(input_tensor=pairs, axis=-1) + 1)/2
import numpy as np
idcg = 15. / np.log(2+np.arange(40))
for i in range(1, len(idcg)):
    idcg[i] = idcg[i] + idcg[i-1]
inverse_idcg = 1. / idcg
def get_ndcg(scores, ratings):
    ranks = get_ranks(scores)
    discounts = 1./ tf.math.log1p(ranks)
    gains = tf.pow(2., tf.nn.relu(ratings))-1
    dcg = tf.reduce_sum(gains * discounts, axis=-1)
    ndcg = dcg * inverse_idcg[10]
    return ndcg