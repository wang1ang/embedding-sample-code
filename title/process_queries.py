import tensorflow as tf
import numpy as np
import data

with open(r'E:\embedding\q.tsv') as f:
    queries = [l.strip() for l in f.readlines()]

root = r'E:\embedding\title\\'
frozen_graph = root + 'title_model_wy_multi.pb'
with tf.gfile.GFile(frozen_graph, "rb") as f:
    restored_graph_def = tf.GraphDef()
    restored_graph_def.ParseFromString(f.read())

tf.import_graph_def(
    restored_graph_def,
    input_map=None,
    return_elements=None,
    name=""
)

graph = tf.get_default_graph()
doc_indices = graph.get_tensor_by_name('doc_indices:0')
doc_values = graph.get_tensor_by_name('doc_values:0')
doc_length = graph.get_tensor_by_name('doc_length:0')
content = graph.get_tensor_by_name('content:0')

with open('')

def test_net(sess, node, feed_dict):
    results = sess.run(node, feed_dict=feed_dict)
    for i, result in enumerate(results):
        print (node[i].name)
        print (result)
        print (np.min(result), np.mean(result), np.max(result))
        print ()
count = 0
acc = 0
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    batches = test_feeder.get_batch_test()
    for i in range(64000):
        sims = []
        for j in range(2):
            sample = next(batches)
            feed_dict = {
                doc_indices : sample['doc_indices'],
                doc_values : sample['doc_values'],
                doc_length : sample['doc_length'],
                content : sample['content'],
                #url : sample['url'],
            }
            test_net(sess, [doc_raw, doc_tan, doc_net, con_net], feed_dict)
            sims.append(sess.run(cosine, feed_dict=feed_dict))
        pos = sims[0]
        neg = sims[1]
        count += 1
        acc += sum(pos > neg)
        if count % 640 == 0:
            print (acc, count, acc / count)
print (acc, count, acc / count)