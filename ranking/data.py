import tensorflow as tf
import numpy as np
import gzip
import random
import math
import os
import io
def dump_common_header(input_filenames, feature_whitelist, header_file):
    inter = set()
    for filename in input_filenames:
        #with gzip.open(filename, 'r') as f:
        with open(filename, 'r', encoding='utf-8') as f:
            line = f.readline().rstrip() #.decode('utf-8')
        header = line.split('\t')
        inter = inter.intersection(set(header)) if len(inter) > 0 else set(header)
    header = inter.difference(['m:RawQuery', 'm:Url', 'm:Rating', 'm:RawRating'])
    #header = [c for c in header if 'BingSAT' in c]
    if feature_whitelist:
        header = [c for c in header if c in feature_whitelist]
    header = sorted([c for c in header if not c.startswith('m:')])
    with open(header_file, 'w') as f:
        f.writelines([h+'\n' for h in header])
    return header
def filter_header(header, key):
    fset1 = [h for h in header if key not in h]
class DataFeeder(object):
    def __init__(self, input_filenames, model_dir, feature_whitelist=None, batch_size=32, result_num = 10):
        # use the first file as test
        self.filenames = input_filenames
        header_file = os.path.join(model_dir, 'selected_header.txt')
        if os.path.isfile(header_file):
            with open(header_file, 'r') as f:
                header = f.read().splitlines()
        else:
            if not os.path.isdir(model_dir):
                os.mkdir(model_dir)
            header = dump_common_header(input_filenames, feature_whitelist, header_file)
        #header = filter_header(header, 'MMVideoBingSATV7enFixedDecayTraining')
        #header = filter_header(header, 'MMVideoCombinedClicksV4enFixedDecayTraining')
        self.header = header
        print ('selected feature size:' + str(len(self.header)))
        self.batch_size = batch_size
        self.result_num = result_num
        self.cache = [[] for i in range(result_num +1)]
        self.get_norm()
        self.train_batch = self.get_batch(self.filenames[1:], self.result_num)
        self.test_batch = self.get_batch(self.filenames[0:1], self.result_num)
    def get_norm(self):
        temp_file = 'feat_stat.npy'
        if os.path.isfile(temp_file):
            moments = np.load(temp_file).item()
        else:
            m = 0
            m2 = 0
            n = 50000
            line_iter = self.get_line(self.filenames[-1:])
            for i in range(n):
                query, url, rating, features = next(line_iter)
                m += features
                m2 += features ** 2
                if i % 10000 == 0:
                    print (i)
            m = m / n
            m2 = m2 / n
            moments = {self.header[i] : [m[i], m2[i]] for i in range(len(self.header))}
            np.save(temp_file, moments)
        m = [moments[field][0] for field in self.header]
        m2 = [moments[field][1] for field in self.header]
        var = m2# - m**2
        self.dev = np.sqrt(var)
    def get_line(self, filenames):
        epoch = -1
        while True:
            epoch += 1
            print ('epoch {0}'.format(epoch))
            for filename in filenames:
                print (f'(epoch {epoch}) reading {filename}')
                #gz = gzip.open(filename, 'r')
                #f = io.BufferedReader(gz)
                with open(filename, 'r', encoding='utf-8') as f:
                    #this_header = f.readline().rstrip().decode('utf-8').split('\t')
                    this_header = f.readline().rstrip().split('\t')
                    order = {this_header[i]:i for i in range(len(this_header))}
                    index = [order[f] for f in self.header]
                    idx_q = order['m:RawQuery']
                    idx_u = order['m:Url']
                    idx_r = order['m:Rating'] if 'm:Rating' in order else order['m:RawRating']
                    for line in f:
                        tokens = line.rstrip().split('\t') #.decode('utf-8')
                        query, url, rating = [tokens[i] for i in [idx_q, idx_u, idx_r]]
                        l = len(tokens)
                        features = [tokens[i] if i < l else '' for i in index]
                        features = [int(f) if f else 0 for f in features]
                        yield query, url, float(rating), np.array(features, dtype=np.float)
                #gz.close()
    @staticmethod
    def transform(ratings):
        return np.log2(np.array(ratings) / 6 + 1)
    def make_batch(self, samples, trans=None):
        query, url, rating, features = [], [], [], []
        for s in samples:
            query.append(s['query'])
            url.append(s['url'])
            rating.append(s['rating'])
            f = s['features']/self.dev
            f = np.minimum(f, 5)
            features.append(f)
        if trans:
            rating = trans(rating)
        return {'query': query, 'url': url, 'rating': rating, 'features': features}
    def get_batch(self, filenames, max_size):
        line_getter = self.get_line(filenames)
        query_prev = ''
        batch = []
        while True:
            query, url, rating, features = next(line_getter)
            if query != query_prev and len(batch) > 0:
                sampled = random.sample(batch, max_size) if max_size < len(batch) else batch
                ratings = [x['rating'] for x in sampled]
                if min(ratings) != max(ratings):
                    yield self.make_batch(sampled)
                batch = []
            query_prev = query
            batch.append({'query': query, 'url': url, 'rating': rating, 'features': features})
    def get_batch_train(self):
        while True:
            size = self.result_num
            while len(self.cache[size]) < self.batch_size:
                batch = next(self.train_batch)
                size = len(batch['rating'])
                self.cache[size].append(batch)
            #print ('train_batches: ', len(self.cache[size]), size)
            yield self.make_batch(self.cache[size], self.transform)
            self.cache[size] = []
    def get_batch_test(self):
        while True:
            batch = next(self.test_batch)
            #print ('test_batches: ', 1, len(batch['rating']))
            yield self.make_batch([batch], self.transform)
    def data_tensor(self, istrain=True):
        with tf.name_scope('data_tensor'):
            dataset = tf.data.Dataset.from_generator(self.get_batch_train if istrain else self.get_batch_test, {
                'query': tf.string,
                'url': tf.string,
                'rating': tf.float32,
                'features': tf.float32
            }, {
                'query': tf.TensorShape([None, None]),
                'url': tf.TensorShape([None, None]),
                'rating': tf.TensorShape([None, None]),
                'features': tf.TensorShape([None, None, len(self.header)]),
            })
            iterator = dataset.make_one_shot_iterator()
            return iterator.get_next()

if __name__ == '__main__':
    files = ['data\\head_{}'.format(i) for i in range(1, 6)]
    data = DataFeeder(files, 'models/bingsat_reg_adam_batch_topfeat')
    batch_iter = data.get_batch_train()
    batch = next(batch_iter)
    print (batch)

