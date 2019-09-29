import tensorflow as tf
from tokenizer.simplifier import Simplifier
from tokenizer.tokenizer import Tokenizer
from tokenizer.lowercase_and_remove_accent import run_strip_accents
import datetime

root = r'D:\embedding\title\\'
#frozen_graph = root + 'title_model_b1_multi.pb'
suffix = ''#'_longer'
frozen_graph = root + 'title_model_b1_multi{}.pb'.format(suffix)

# en-us ranker
#infile = r'D:\temp\JuneRanker\q.tsv'
#outfile = r'D:\temp\JuneRanker\content_b1_multi_query.tsv'.format(suffix)

# gdi ranker
infile = r'D:\temp\JuneRanker\gdi_q.tsv'
outfile = r'D:\temp\JuneRanker\gdi_content_b1_multi_query.tsv'.format(suffix)


# everest 05 & 06
#infile = r'D:\temp\JuneRanker\QuerySet-Combined.tsv'
#outfile = r'D:\temp\JuneRanker\QuerySet-Combined_xlm{}.tsv'.format(suffix)

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
doc_ids = graph.get_tensor_by_name('doc_ids:0')
doc_mask = graph.get_tensor_by_name('doc_mask:0')
doc_type = graph.get_tensor_by_name('doc_type:0')
#content = graph.get_tensor_by_name('content:0')
doc_output = graph.get_tensor_by_name('doc/output:0')

doc_max_length = 12


s = Simplifier('tokenizer/zh_mapping.txt')
t = Tokenizer('tokenizer/spiece_all_bpe/spiece.all.bpe.130000.lower.model',
    'tokenizer/lg.all.voc',
    doc_max_length
)
count = 0
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
#f = ['a real test', 'b false test']
with open(infile, 'r', encoding='utf-8') as f, open(outfile, 'w', encoding='utf-8') as fo:
    with tf.Session(config=config) as sess:
        time = datetime.datetime.now()
        for line in f:
            simple = s.simplify(line)
            tokens = t.tokenize(simple)
            accents = run_strip_accents(tokens)
            ids = t.token_to_id(accents)

            l = len(ids)
            _ids  = ids + [0] * (doc_max_length - l)
            _mask = [1] * l + [0] * (doc_max_length - l)
            _type = [0] * doc_max_length
            feed_dict = {
                doc_ids: [_ids],
                doc_mask: [_mask],
                doc_type: [_type],
            }
            result = sess.run(doc_output, feed_dict=feed_dict)
            result = [str(int(round(r * 65536))) for r in result[0]]
            fo.write(line.rstrip() + '\t' + '\t'.join(result))
            fo.write('\n')
            count += 1
            if count % 2000 == 0:
                temp = datetime.datetime.now()
                print (count, (temp-time).total_seconds())
                time = temp
                #break
print (count)