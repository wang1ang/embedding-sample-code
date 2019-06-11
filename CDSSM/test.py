import tensorflow as tf
import numpy as np
import data_prepare
import os
#model_name = r'model_test\v14_big\model.ckpt-960307.meta'
model_name = r'model_test\v15_64\model.ckpt-1000000.meta'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    # load graph
    new_saver = tf.train.import_meta_graph(model_name)
    # load values
    new_saver.restore(sess, tf.train.latest_checkpoint(os.path.dirname(model_name)))
    
    # nodes
    graph = tf.get_default_graph()
    doc_indices = graph.get_tensor_by_name('sparse_data_tensor/IteratorGetNext:0')
    doc_length = graph.get_tensor_by_name('sparse_data_tensor/IteratorGetNext:1')
    doc_values = graph.get_tensor_by_name('sparse_data_tensor/IteratorGetNext:2')
    input_indices = graph.get_tensor_by_name('sparse_data_tensor/IteratorGetNext:3')
    input_length = graph.get_tensor_by_name('sparse_data_tensor/IteratorGetNext:4')
    input_values = graph.get_tensor_by_name('sparse_data_tensor/IteratorGetNext:5')
    query = graph.get_tensor_by_name('sparse_data_tensor/IteratorGetNext:6')
    url = graph.get_tensor_by_name('sparse_data_tensor/IteratorGetNext:7')
    cosine = graph.get_tensor_by_name('self_cosine_loss/DiagPart_1:0')

    # load data
    vocab = data_prepare.trilettergram()
    data_feeder = data_prepare.DataFeeder(
        128,
        '../Processed_ascii_sample_eval_shuffle_URL_fx.tsv',
        1, 3, 1, 2,
        vocab, 16, 64,
        skip_count=0,
        training = False,
        use_lgram=False)

    count = 0
    acc = 0
    for batch in data_feeder.get_batch_test():
        feed_dict = {
            doc_indices : batch['doc_indices'],
            doc_length : batch['doc_length'],
            doc_values : batch['doc_values'],
            input_indices : batch['input_indices'],
            input_length : batch['input_length'],
            input_values : batch['input_values'],
            query : batch['query'],
            url : batch['url'],
        }
        sims = sess.run(cosine, feed_dict=feed_dict)
        pos = sims[0::2]
        neg = sims[1::2]
        count += len(pos)
        acc += sum(pos > neg)
        if count % 640 == 0:
            print (acc, count, acc / count)
        if count == 64000:
            break
    print (acc, count, acc / count)