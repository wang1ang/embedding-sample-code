import tensorflow as tf
import os
import numpy as np
import sys
sys.path.append('../')
from data import SparseDataFeeder, TriLetterEmbedding

version = '6.1'
checkpoint_dir  = r'E:\embedding\title\models\v{}'.format(version) #model.ckpt-1057926'
ofile = 'v{}_sample.txt'.format(version)
os.environ['CUDA_VISIBLE_DEVICES'] = ''

config = tf.ConfigProto()
# occupy gpu gracefully
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    # load graph
    model = tf.train.latest_checkpoint(checkpoint_dir)
    # option 1: from meta file
    saver = tf.train.import_meta_graph(model+'.meta', clear_devices=True)
    
    # option 2: from predefined net
    #doc_net, con_net, cosine = model(get_params())
    #saver = tf.train.Saver()

    # load values
    saver.restore(sess, model) #os.path.dirname(checkpoint)))

    graph = tf.get_default_graph()

    content = graph.get_tensor_by_name('sparse_data_tensor/IteratorGetNext:0')
    doc_indices = graph.get_tensor_by_name('sparse_data_tensor/IteratorGetNext:1')
    doc_values = graph.get_tensor_by_name('sparse_data_tensor/IteratorGetNext:3')
    doc_length = graph.get_tensor_by_name('sparse_data_tensor/IteratorGetNext:2')
    
    #scale = graph.get_tensor_by_name('self_cosine_loss/weight_1:0')
    #bias = graph.get_tensor_by_name('self_cosine_loss/b_1:0')
    #print ('scale:', sess.run(scale))
    #print ('bias:', sess.run(bias))
    
    # image = tf.placeholder(tf.float32,[1,192,192,3])
    # print(image)
    # output = graph.get_tensor_by_name('self_cosine_loss/add:0')
    output = graph.get_tensor_by_name('self_cosine_loss/Softmax:0')


    test_feeder = SparseDataFeeder(
        128,
        [['../train_HRS_title_April.txt']],
        64,
        skip_count=0,
        embed_func=TriLetterEmbedding,
        training = False
    )
    batch = next(test_feeder.poll_batch())
    feed_dict = {
        content: batch['content'], #np.zeros([128, 2048], np.float),
        doc_indices: batch['doc_indices'], #np.array([[i*64, 0] for i in range(129)], np.int64),
        doc_values: batch['doc_values'], #np.array([i * 5 + 3 for i in range(129)], np.int32),
        doc_length: batch['doc_length'], #np.array([1]*128, np.int32)
    }
    result = sess.run(output, feed_dict = feed_dict)
    np.savetxt(ofile, result, delimiter='\t')
    print (ofile)