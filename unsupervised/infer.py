import tensorflow as tf
import numpy as np
from datafeeder import DataFeeder

config = tf.ConfigProto()
# occupy gpu gracefully
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    # load graph
    new_saver = tf.train.import_meta_graph('log3/model.ckpt-1010600.meta')
    # load values
    new_saver.restore(sess, tf.train.latest_checkpoint('log3/'))
    
    # nodes
    graph = tf.get_default_graph()
    content_tensor = graph.get_tensor_by_name('IteratorGetNext:2')
    dan_tensor = graph.get_tensor_by_name('fc/fc_dan/BiasAdd:0')
    cdssm_tensor = graph.get_tensor_by_name('fc/fc_cdssm/BiasAdd:0')

    feeder = DataFeeder('train_hrs_video.txt')
    with open('pred_v3.txt', 'w') as f:
        for i, (key, dan, cdssm, content) in enumerate(feeder.read_feature()):
            dan = np.array(dan).astype(int)
            cdssm = np.array(cdssm).astype(int)
            content = np.expand_dims(content, 0)
            dan_v, cdssm_v = sess.run([dan_tensor, cdssm_tensor], feed_dict={content_tensor:content})
            dan_v = np.round(np.minimum(255, np.maximum(0, dan_v[0]))).astype(int)
            cdssm_v = np.round(np.minimum(255, np.maximum(0, cdssm_v[0]))).astype(int)
            f.write(key + '\t' + '\t'.join(str(x) for x in dan_v) + '\t' + '\t'.join(str(x) for x in cdssm_v) + '\n')
            if i % 10000 == 0:
                print (i, dan[:3], dan_v[:3], cdssm[:3], cdssm_v[:3])
